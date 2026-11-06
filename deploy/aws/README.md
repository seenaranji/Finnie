# Deploying Finnie to AWS (Step 29)

```
internet -> ALB (80/443, only allowed_cidrs) -> ECS Fargate task [ui:8501 + api:8000 share localhost] -> RDS PostgreSQL 17 + pgvector (private subnets)
```
Secrets (database URL, JWT secret, API keys) are in Secrets Manager and injected at start. Logs go to CloudWatch `/finnie`.
Only the UI is exposed; the API is reachable only from the UI container.

## Estimated monthly cost (us-east-1, running 24/7; verify on the AWS pricing pages)
| Item | ~USD |
|---|---|
| Fargate 1 vCPU / 2 GB | 36 |
| RDS db.t4g.micro + 20 GB gp3 | 15 |
| Application Load Balancer | 17 |
| Secrets Manager (4 secrets) | 1.6 |
| ECR, CloudWatch logs, data | 1-3 |
| **Total** | **~70** (about $2.30/day). No NAT gateway (would add ~$33). |

`terraform destroy` when you are done testing.

## Steps (nothing here has been run yet)
Prereqs: `aws configure` (or SSO), Docker, Terraform. Check: `aws sts get-caller-identity`.

1. Configure
   ```
   cd deploy/aws
   cp terraform.tfvars.example terraform.tfvars      # set allowed_cidrs to your IP/32
   export TF_VAR_openai_api_key=sk-...
   export TF_VAR_alpha_vantage_api_key=...           # optional
   ```
2. Create the registry first, so there is somewhere to push the image
   ```
   terraform init
   terraform apply -target=aws_ecr_repository.app
   ```
3. Build for AWS (Fargate here is x86; on an Apple-silicon Mac you must say so) and push
   ```
   REPO=$(terraform output -raw ecr_repository)
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${REPO%%/*}
   docker build --platform linux/amd64 -t $REPO:latest ../..
   docker push $REPO:latest
   ```
4. Create everything else (about 10 min, the database is slowest)
   ```
   terraform plan      # read it
   terraform apply
   ```
5. Create the tables, then fill the knowledge base (one-off tasks using the same image)
   ```
   run() { aws ecs run-task --cluster finnie --launch-type FARGATE --task-definition finnie-job \
     --network-configuration "awsvpcConfiguration={subnets=[$(terraform output -json public_subnet_ids | jq -r 'join(",")')],securityGroups=[$(terraform output -raw task_security_group)],assignPublicIp=ENABLED}" \
     --overrides "{\"containerOverrides\":[{\"name\":\"job\",\"command\":$1}]}" --query 'tasks[0].taskArn' --output text; }
   run '["alembic","upgrade","head"]'
   # wait until it stops (watch /finnie logs, stream prefix job/), then:
   run '["python","-m","scripts.index_knowledge_base"]'
   ```
   The second one costs about $0.001 of embeddings.
6. Open `terraform output url`, register an account, ask a question.
7. Look: `aws logs tail /finnie --follow`; `aws ecs describe-services --cluster finnie --services finnie`.
8. Update the app later: build and push a new image, then `aws ecs update-service --cluster finnie --service finnie --force-new-deployment`.
9. Tear down: `terraform destroy`.

## Known limits
- HTTP only unless `certificate_arn` is set (needs a domain + ACM certificate). Over HTTP, passwords and tokens are readable on the network: use only from your own IP for a test.
- One task, no autoscaling, single-AZ database (set `multi_az` for real use).
- Terraform state contains the generated secrets; it is git-ignored, but treat the file as sensitive.
- `.alpha_usage.json` resets whenever the task restarts (same as in Docker).
- Task has no IAM role of its own (it calls no AWS services).
