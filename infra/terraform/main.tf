locals {
  tags             = merge(var.tags, { Application = "GridCast", ManagedBy = "Terraform" })
  model_bucket_arn = aws_s3_bucket.model_registry.arn
  model_uri        = "s3://${aws_s3_bucket.model_registry.bucket}/${var.model_prefix}${var.model_version}"
  adot_config      = <<-YAML
    receivers:
      prometheus:
        config:
          scrape_configs:
            - job_name: gridcast
              scrape_interval: 15s
              static_configs: [{ targets: ["127.0.0.1:8080"] }]
    exporters:
      awsemf:
        namespace: GridCast/Serving
        log_group_name: ${aws_cloudwatch_log_group.adot.name}
        log_stream_name: '{TaskId}'
        dimension_rollup_option: NoDimensionRollup
        metric_declarations:
          - dimensions: [[reason]]
            metric_name_selectors: ["^gridcast_fallbacks_total$"]
    service:
      pipelines:
        metrics:
          receivers: [prometheus]
          exporters: [awsemf]
  YAML
}

resource "aws_ecr_repository" "serving" {
  name                 = var.name
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.model_registry.arn
  }
  tags = local.tags
}

resource "aws_ecr_lifecycle_policy" "serving" {
  repository = aws_ecr_repository.serving.name
  policy     = jsonencode({ rules = [{ rulePriority = 1, description = "Keep the newest 30 immutable images", selection = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 30 }, action = { type = "expire" } }] })
}

resource "aws_kms_key" "model_registry" {
  description             = "GridCast model registry encryption"
  enable_key_rotation     = true
  deletion_window_in_days = 30
  policy                  = jsonencode({ Version = "2012-10-17", Statement = [{ Sid = "EnableRootAccountAdministration", Effect = "Allow", Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }, Action = "kms:*", Resource = "*" }] })
  tags                    = local.tags
}
resource "aws_kms_alias" "model_registry" {
  name          = "alias/${var.name}-models"
  target_key_id = aws_kms_key.model_registry.key_id
}

# Object Lock can only be enabled when the bucket is first created; toggling it later
# requires bucket replacement and is deliberately an explicit deployment decision.
resource "aws_s3_bucket" "model_registry" {
  #checkov:skip=CKV2_AWS_62:Registry changes are deployed explicitly; event destination is account-specific.
  #checkov:skip=CKV_AWS_18:Central S3 access logging bucket is intentionally supplied by the consuming account.
  #checkov:skip=CKV_AWS_144:Cross-region replication is a data-residency/DR choice outside this single-region reference.
  bucket              = "${var.name}-${data.aws_caller_identity.current.account_id}-${var.region}"
  object_lock_enabled = var.enable_object_lock
  tags                = local.tags
}
data "aws_caller_identity" "current" {}
resource "aws_s3_bucket_versioning" "model_registry" {
  bucket = aws_s3_bucket.model_registry.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_server_side_encryption_configuration" "model_registry" {
  bucket = aws_s3_bucket.model_registry.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.model_registry.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}
resource "aws_s3_bucket_public_access_block" "model_registry" {
  bucket                  = aws_s3_bucket.model_registry.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_lifecycle_configuration" "model_registry" {
  bucket = aws_s3_bucket.model_registry.id
  rule {
    id     = "expire-old-model-versions"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload { days_after_initiation = 7 }
    noncurrent_version_expiration { noncurrent_days = 90 }
  }
}
resource "aws_s3_bucket_policy" "tls_only" {
  bucket = aws_s3_bucket.model_registry.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{ Sid = "DenyInsecureTransport", Effect = "Deny", Principal = "*", Action = "s3:*", Resource = [local.model_bucket_arn, "${local.model_bucket_arn}/*"], Condition = { Bool = { "aws:SecureTransport" = "false" } } }] })
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.name}"
  retention_in_days = 365
  kms_key_id        = aws_kms_key.model_registry.arn
  tags              = local.tags
}
resource "aws_cloudwatch_log_group" "adot" {
  name              = "/ecs/${var.name}/adot"
  retention_in_days = 365
  kms_key_id        = aws_kms_key.model_registry.arn
  tags              = local.tags
}
resource "aws_ecs_cluster" "this" {
  name = var.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = local.tags
}

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}
resource "aws_iam_role" "task_execution" {
  name               = "${var.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = local.tags
}
resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
resource "aws_iam_role" "task" {
  name               = "${var.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = local.tags
}
resource "aws_iam_role_policy" "model_read" {
  name = "read-model-prefix"
  role = aws_iam_role.task.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = local.model_bucket_arn, Condition = { StringLike = { "s3:prefix" = ["${var.model_prefix}*"] } } },
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${local.model_bucket_arn}/${var.model_prefix}*" },
    { Effect = "Allow", Action = ["kms:Decrypt"], Resource = aws_kms_key.model_registry.arn }
  ] })
}

resource "aws_security_group" "alb" {
  name        = "${var.name}-alb"
  vpc_id      = var.vpc_id
  description = "Public HTTPS ingress for GridCast"
  ingress {
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS"
  }
  ingress {
    protocol    = "tcp"
    from_port   = 8443
    to_port     = 8443
    cidr_blocks = ["0.0.0.0/0"]
    description = "CodeDeploy test listener; restrict in a real deployment"
  }
  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow outbound connections to AWS services"
  }
  tags = local.tags
}
resource "aws_security_group" "task" {
  name        = "${var.name}-task"
  vpc_id      = var.vpc_id
  description = "ALB-only GridCast task ingress"
  ingress {
    protocol        = "tcp"
    from_port       = 8080
    to_port         = 8080
    security_groups = [aws_security_group.alb.id]
    description     = "Allow ALB traffic to the serving port"
  }
  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow task egress to model registry and telemetry"
  }
  tags = local.tags
}
resource "aws_lb" "this" {
  #checkov:skip=CKV_AWS_91:ALB access-log bucket and its cross-account delivery policy are account-level resources.
  #checkov:skip=CKV2_AWS_28:WAF association is optional and must use an account-approved Web ACL, documented in README.
  name                       = substr(var.name, 0, 32)
  internal                   = false
  load_balancer_type         = "application"
  security_groups            = [aws_security_group.alb.id]
  subnets                    = var.public_subnet_ids
  drop_invalid_header_fields = true
  enable_deletion_protection = true
  desync_mitigation_mode     = "strictest"
  tags                       = local.tags
}
resource "aws_lb_target_group" "blue" {
  #checkov:skip=CKV_AWS_378:TLS terminates at the HTTPS ALB; private task ENI traffic remains within VPC.
  name                 = substr("${var.name}-blue", 0, 32)
  port                 = 8080
  protocol             = "HTTP"
  vpc_id               = var.vpc_id
  target_type          = "ip"
  deregistration_delay = 30
  health_check {
    path                = "/ready"
    matcher             = "200"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 15
  }
  tags = local.tags
}
resource "aws_lb_target_group" "green" {
  #checkov:skip=CKV_AWS_378:TLS terminates at the HTTPS ALB; private task ENI traffic remains within VPC.
  name                 = substr("${var.name}-green", 0, 32)
  port                 = 8080
  protocol             = "HTTP"
  vpc_id               = var.vpc_id
  target_type          = "ip"
  deregistration_delay = 30
  health_check {
    path                = "/ready"
    matcher             = "200"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 15
  }
  tags = local.tags
}
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.this.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.blue.arn
  }
}
resource "aws_lb_listener" "test" {
  load_balancer_arn = aws_lb.this.arn
  port              = 8443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.green.arn
  }
}

resource "aws_ecs_task_definition" "serving" {
  family                   = var.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn
  # Fargate has no tmpfs option. This empty task volume is ephemeral and is mounted
  # at /tmp so the image can retain a read-only root filesystem.
  volume { name = "tmp" }
  container_definitions = jsonencode([
    { name = "serving", image = var.image, essential = true, user = "gridcast", readonlyRootFilesystem = true, stopTimeout = 20, portMappings = [{ containerPort = 8080, protocol = "tcp" }], mountPoints = [{ sourceVolume = "tmp", containerPath = "/tmp", readOnly = false }], environment = [{ name = "GRIDCAST_MODEL_URI", value = local.model_uri }, { name = "GRIDCAST_ENV", value = "production" }, { name = "GRIDCAST_MAX_IN_FLIGHT", value = "8" }, { name = "GRIDCAST_MAX_ADMITTED", value = "16" }], secrets = [{ name = "GRIDCAST_PLACEHOLDER_SECRET", valueFrom = var.placeholder_secret_arn }], logConfiguration = { logDriver = "awslogs", options = { awslogs-group = aws_cloudwatch_log_group.app.name, awslogs-region = var.region, awslogs-stream-prefix = "serving" } }, healthCheck = { command = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=2)\""], interval = 30, timeout = 5, retries = 3, startPeriod = 30 } },
    { name = "adot", image = "public.ecr.aws/aws-observability/aws-otel-collector:v0.43.0", essential = false, environment = [{ name = "AOT_CONFIG_CONTENT", value = local.adot_config }], command = ["--config=env:AOT_CONFIG_CONTENT"], logConfiguration = { logDriver = "awslogs", options = { awslogs-group = aws_cloudwatch_log_group.adot.name, awslogs-region = var.region, awslogs-stream-prefix = "adot" } } }
  ])
  tags = local.tags
}
resource "aws_ecs_service" "serving" {
  name                              = var.name
  cluster                           = aws_ecs_cluster.this.id
  task_definition                   = aws_ecs_task_definition.serving.arn
  desired_count                     = var.desired_count
  launch_type                       = "FARGATE"
  health_check_grace_period_seconds = 60
  deployment_controller { type = "CODE_DEPLOY" }
  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = false
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.blue.arn
    container_name   = "serving"
    container_port   = 8080
  }
  lifecycle { ignore_changes = [task_definition, desired_count, load_balancer] }
  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "target_5xx" {
  alarm_name          = "${var.name}-target-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 1
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Sum"
  dimensions          = { LoadBalancer = aws_lb.this.arn_suffix, TargetGroup = aws_lb_target_group.blue.arn_suffix }
  treat_missing_data  = "notBreaching"
}
resource "aws_cloudwatch_metric_alarm" "target_p95" {
  alarm_name          = "${var.name}-target-p95"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  threshold           = 0.5
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  extended_statistic  = "p95"
  dimensions          = { LoadBalancer = aws_lb.this.arn_suffix, TargetGroup = aws_lb_target_group.blue.arn_suffix }
  treat_missing_data  = "notBreaching"
}
resource "aws_cloudwatch_metric_alarm" "fallback_rate" {
  alarm_name          = "${var.name}-fallback-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 0.05
  datapoints_to_alarm = 2
  treat_missing_data  = "notBreaching"
  metric_query {
    id = "fallbacks"
    metric {
      metric_name = "gridcast_fallbacks_total"
      namespace   = "GridCast/Serving"
      period      = 60
      stat        = "Sum"
    }
    return_data = false
  }
  metric_query {
    id = "requests"
    metric {
      metric_name = "gridcast_predictions_total"
      namespace   = "GridCast/Serving"
      period      = 60
      stat        = "Sum"
    }
    return_data = false
  }
  metric_query {
    id          = "rate"
    expression  = "100 * fallbacks / MAX([requests, 1])"
    label       = "Fallback percentage"
    return_data = true
  }
}
resource "aws_codedeploy_app" "serving" {
  compute_platform = "ECS"
  name             = var.name
}
resource "aws_iam_role" "codedeploy" {
  name               = "${var.name}-codedeploy"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "codedeploy.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}
resource "aws_iam_role_policy_attachment" "codedeploy" {
  role       = aws_iam_role.codedeploy.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSCodeDeployRoleForECS"
}
resource "aws_codedeploy_deployment_group" "serving" {
  app_name               = aws_codedeploy_app.serving.name
  deployment_group_name  = var.name
  service_role_arn       = aws_iam_role.codedeploy.arn
  deployment_config_name = "CodeDeployDefault.ECSCanary10Percent5Minutes"
  deployment_style {
    deployment_type   = "BLUE_GREEN"
    deployment_option = "WITH_TRAFFIC_CONTROL"
  }
  auto_rollback_configuration {
    enabled = true
    events  = ["DEPLOYMENT_FAILURE", "DEPLOYMENT_STOP_ON_ALARM"]
  }
  alarm_configuration {
    enabled = true
    alarms  = [aws_cloudwatch_metric_alarm.target_5xx.alarm_name, aws_cloudwatch_metric_alarm.target_p95.alarm_name, aws_cloudwatch_metric_alarm.fallback_rate.alarm_name]
  }
  blue_green_deployment_config {
    terminate_blue_instances_on_deployment_success {
      action                           = "TERMINATE"
      termination_wait_time_in_minutes = 5
    }
    deployment_ready_option { action_on_timeout = "CONTINUE_DEPLOYMENT" }
  }
  ecs_service {
    cluster_name = aws_ecs_cluster.this.name
    service_name = aws_ecs_service.serving.name
  }
  load_balancer_info {
    target_group_pair_info {
      prod_traffic_route { listener_arns = [aws_lb_listener.https.arn] }
      test_traffic_route { listener_arns = [aws_lb_listener.test.arn] }
      target_group { name = aws_lb_target_group.blue.name }
      target_group { name = aws_lb_target_group.green.name }
    }
  }
}

resource "aws_appautoscaling_target" "service" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.serving.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}
resource "aws_appautoscaling_policy" "requests" {
  name               = "alb-requests"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.service.resource_id
  scalable_dimension = aws_appautoscaling_target.service.scalable_dimension
  service_namespace  = aws_appautoscaling_target.service.service_namespace
  target_tracking_scaling_policy_configuration {
    target_value = 100
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.this.arn_suffix}/${aws_lb_target_group.blue.arn_suffix}"
    }
  }
}
resource "aws_appautoscaling_policy" "cpu" {
  name               = "cpu-60"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.service.resource_id
  scalable_dimension = aws_appautoscaling_target.service.scalable_dimension
  service_namespace  = aws_appautoscaling_target.service.service_namespace
  target_tracking_scaling_policy_configuration {
    target_value = 60
    predefined_metric_specification { predefined_metric_type = "ECSServiceAverageCPUUtilization" }
  }
}

data "tls_certificate" "github_actions" { url = "https://token.actions.githubusercontent.com" }
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github_actions.certificates[0].sha1_fingerprint]
}
data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:gioalvari/gridcast:ref:refs/tags/v*",
        "repo:gioalvari/gridcast:environment:production",
      ]
    }
  }
}
resource "aws_iam_role" "github_deploy" {
  name               = "${var.name}-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_assume.json
  tags               = local.tags
}
resource "aws_iam_role_policy" "github_deploy" {
  #checkov:skip=CKV_AWS_355:ECR authorization and task-definition registration are AWS APIs that do not support resource ARNs.
  #checkov:skip=CKV_AWS_290:Write permissions are limited by GitHub tag/environment OIDC trust and named ECR/CodeDeploy/IAM roles.
  name   = "deploy-serving"
  role   = aws_iam_role.github_deploy.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" }, { Effect = "Allow", Action = ["ecr:BatchCheckLayerAvailability", "ecr:CompleteLayerUpload", "ecr:InitiateLayerUpload", "ecr:PutImage", "ecr:UploadLayerPart"], Resource = aws_ecr_repository.serving.arn }, { Effect = "Allow", Action = ["ecs:RegisterTaskDefinition", "ecs:DescribeServices", "ecs:UpdateService"], Resource = "*" }, { Effect = "Allow", Action = ["codedeploy:CreateDeployment"], Resource = aws_codedeploy_deployment_group.serving.arn }, { Effect = "Allow", Action = ["iam:PassRole"], Resource = [aws_iam_role.task.arn, aws_iam_role.task_execution.arn] }] })
}
