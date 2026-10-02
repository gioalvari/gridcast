output "ecr_repository_url" {
  value = aws_ecr_repository.serving.repository_url
}
output "model_registry_bucket" {
  value = aws_s3_bucket.model_registry.bucket
}
output "load_balancer_dns_name" {
  value = aws_lb.this.dns_name
}
output "github_deploy_role_arn" {
  value = aws_iam_role.github_deploy.arn
}
