Configure Repository Secrets
Go to GitHub → Repository → Settings → Secrets and variables → Actions

Add these secrets:

DOCKER_USERNAME - Docker Hub username (if using)
DOCKER_PASSWORD - Docker Hub password (if using)
AZURE_CREDENTIALS - Azure credentials (for Azure deployment)
AWS_ACCESS_KEY_ID - AWS access key (for AWS deployment)
AWS_SECRET_ACCESS_KEY - AWS secret key (for AWS deployment)
Step 4: Enable GitHub Actions
Go to GitHub → Repository → Actions and enable workflows

Step 5: Configure Branch Protection
Go to GitHub → Repository → Settings → Branches

Add branch protection rule for main:

✅ Require a pull request before merging
✅ Require status checks to pass before merging
✅ Require branches to be up to date before merging



Azure Services Used:

Service	Purpose
Azure Container Registry	Store Docker images
Azure Container Apps	Host API containers
Azure Blob Storage	Store DVC data and models
Azure Log Analytics	Centralized logging
Azure Application Insights	API monitoring and telemetry


┌─────────────────────────────────────────────────────────────────────┐
│                     AZURE ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐     ┌─────────────────────┐
│   GitHub Actions    │────▶│  Azure Container    │
│   (CI/CD Pipeline)  │     │  Registry (ACR)     │
└─────────────────────┘     └──────────┬──────────┘
                                       │
                                       ▼
┌─────────────────────┐     ┌─────────────────────┐
│   Azure Blob        │     │  Azure Container    │
│   Storage (DVC)     │     │  Apps / App Service │
└─────────────────────┘     └──────────┬──────────┘
                                       │
                                       ▼
┌─────────────────────┐     ┌─────────────────────┐
│   Azure Monitor     │◀────│   Application       │
│   (Logs & Metrics)  │     │   Insights          │
└─────────────────────┘     └─────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    AZURE DEPLOYMENT FLOW                         │
└─────────────────────────────────────────────────────────────────┘

  Developer                  GitHub                    Azure
     │                         │                         │
     │  git push              │                         │
     ├────────────────────────▶                         │
     │                         │                         │
     │                    ┌────┴────┐                   │
     │                    │   CI    │                   │
     │                    │  Tests  │                   │
     │                    └────┬────┘                   │
     │                         │ Pass                   │
     │                    ┌────┴────┐                   │
     │                    │  Build  │                   │
     │                    │  Image  │                   │
     │                    └────┬────┘                   │
     │                         │                         │
     │                         │  Push to ACR           │
     │                         ├────────────────────────▶
     │                         │                         │
     │                         │  Deploy to Container   │
     │                         │  Apps (Staging)        │
     │                         ├────────────────────────▶
     │                         │                         │
     │                         │  Smoke Tests           │
     │                         ├────────────────────────▶
     │                         │                         │
     │  git tag v1.0.0        │                         │
     ├────────────────────────▶                         │
     │                         │                         │
     │                         │  Deploy to Container   │
     │                         │  Apps (Production)     │
     │                         ├────────────────────────▶
     │                         │                         │
     │                         │           ┌────────────┤
     │                         │           │ Application│
     │                         │           │  Insights  │
     │                         │           │ Monitoring │
     │                         │           └────────────┤






     ┌─────────────────────────────────────────────────────────────────────┐
│                    MONITORING ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Production     │     │   Monitoring    │     │   Alert         │
│  API Requests   │────▶│   Service       │────▶│   System        │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
            ┌───────────┐ ┌───────────┐ ┌───────────┐
            │   Data    │ │   Model   │ │  Feature  │
            │   Drift   │ │   Perf.   │ │   Stats   │
            │  Monitor  │ │  Monitor  │ │  Monitor  │
            └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
                  │             │             │
                  └─────────────┼─────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Monitoring Reports  │
                    │   & Dashboards        │
                    └───────────────────────┘



                    ┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLETE MONITORING SYSTEM                        │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Prediction   │     │    Drift       │     │   Performance  │
│    Logger      │────▶│   Detector     │────▶│    Monitor     │
│ (Every Request)│     │   (Evidently)  │     │  (Scheduled)   │
└────────────────┘     └────────────────┘     └────────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │   Alert Manager    │
                    │  ┌──────────────┐  │
                    │  │    Slack     │  │
                    │  │    Email     │  │
                    │  │    Teams     │  │
                    │  └──────────────┘  │
                    └────────────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │     Dashboard      │
                    │  (HTML / Grafana)  │
                    └────────────────────┘