variable "region" {
  type    = string
  default = "eu-central-1"
}
variable "name" {
  type    = string
  default = "gridcast-serving"
}
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "public_subnet_ids" { type = list(string) }
variable "acm_certificate_arn" { type = string }
variable "image" {
  type        = string
  description = "Immutable serving image URI, preferably digest-pinned."
}
variable "model_prefix" {
  type    = string
  default = "models/"
}
variable "model_version" {
  type    = string
  default = "current"
}
variable "task_cpu" {
  type    = number
  default = 1024
}
variable "task_memory" {
  type    = number
  default = 2048
}
variable "desired_count" {
  type    = number
  default = 2
}
variable "min_capacity" {
  type    = number
  default = 2
}
variable "max_capacity" {
  type    = number
  default = 10
}
variable "placeholder_secret_arn" {
  type        = string
  description = "ARN of a pre-created Secrets Manager secret containing a referenced API value; no secret value is managed here."
}
variable "enable_object_lock" {
  type    = bool
  default = false
}
variable "tags" {
  type    = map(string)
  default = {}
}
