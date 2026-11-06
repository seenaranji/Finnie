variable "region" {
  type    = string
  default = "us-east-1"
}

variable "allowed_cidrs" {
  description = "Who may reach the site, e.g. [\"203.0.113.7/32\"] (your own IP). No default on purpose: you must choose. Use [\"0.0.0.0/0\"] only with HTTPS (certificate_arn)."
  type        = list(string)
}

variable "certificate_arn" {
  description = "ACM certificate for HTTPS. Empty = plain HTTP only (passwords and tokens travel unencrypted: fine for a short test from your own IP, not for real use)."
  type        = string
  default     = ""
}

variable "openai_api_key" {
  type      = string
  sensitive = true
}

variable "alpha_vantage_api_key" {
  type      = string
  sensitive = true
  default   = ""
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "task_cpu" {
  type    = number
  default = 1024 # 1 vCPU
}

variable "task_memory" {
  type    = number
  default = 2048 # MiB; measured idle use is ~230 MiB (api + ui), the rest is headroom
}

variable "deletion_protection" {
  description = "Set true for anything you care about. False so `terraform destroy` works for the learning setup."
  type        = bool
  default     = false
}
