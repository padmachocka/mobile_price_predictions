pipeline {
  agent any
  options { timestamps() }

  parameters {
    string(name: 'AWS_REGION',       defaultValue: 'eu-west-2')
    string(name: 'ACCOUNT_ID',       defaultValue: '430006376054')
    string(name: 'ECR_REPO',         defaultValue: 'mobile_price_pred_app')
    string(name: 'EC2_HOST',         defaultValue: '18.175.251.68')
    string(name: 'EC2_USER',         defaultValue: 'ubuntu')
    string(name: 'HOST_PORT',        defaultValue: '8000', description: 'Public port on EC2')
    string(name: 'CONTAINER_PORT',   defaultValue: '8000', description: 'Uvicorn port in container')
    string(name: 'CONTAINER_NAME',   defaultValue: 'mobile-price-app')
    string(name: 'MODEL_FILE',       defaultValue: 'best_xgb_pipeline.pkl', description: 'Pipeline artifact filename created by training')
  }

  environment {
    ECR_URI   = "${params.ACCOUNT_ID}.dkr.ecr.${params.AWS_REGION}.amazonaws.com/${params.ECR_REPO}"
    IMAGE_TAG = "build-${env.BUILD_NUMBER}"
    IMAGE     = "${ECR_URI}:${IMAGE_TAG}"
  }

  stages {

    stage('Checkout') {
      steps {
        checkout scm
        sh 'git log -1 --oneline || true'
      }
    }

    stage('Train & export model') {
      steps {
        sh """
          set -e
          python3 -m venv .venv
          . .venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt
          # Runs pipeline.py which must save the pipeline artifact at repo root:
          #   with open("${params.MODEL_FILE}", "wb") as f: pickle.dump(best_pipe, f)
          python pipeline.py
          test -f ${params.MODEL_FILE}
          ls -lh ${params.MODEL_FILE}
        """
      }
      post { always { sh 'rm -rf .venv || true' } }
    }

    stage('Docker build') {
      steps {
        sh """
          set -e
          # Ensure the model file is not ignored by .dockerignore
          test -f ${params.MODEL_FILE}
          docker build -t ${IMAGE} .
        """
      }
    }

    stage('Login to ECR & Ensure repo') {
      steps {
        withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-creds']]) {
          sh """
            set -e
            aws ecr describe-repositories --repository-names ${params.ECR_REPO} --region ${params.AWS_REGION} \
              || aws ecr create-repository --repository-name ${params.ECR_REPO} --region ${params.AWS_REGION} >/dev/null

            aws ecr get-login-password --region ${params.AWS_REGION} \
              | docker login --username AWS --password-stdin ${ECR_URI}
          """
        }
      }
    }

    stage('Push to ECR') {
      steps {
        sh """
          set -e
          docker push ${IMAGE}
          docker tag ${IMAGE} ${ECR_URI}:latest
          docker push ${ECR_URI}:latest
        """
      }
    }

    stage('Deploy to EC2') {
      steps {
        sshagent(credentials: ['ec2-ssh-key']) {
          sh """
            set -e
            ssh -o StrictHostKeyChecking=no ${params.EC2_USER}@${params.EC2_HOST} '
              set -e
              aws ecr get-login-password --region ${params.AWS_REGION} \
                | sudo docker login --username AWS --password-stdin ${params.ACCOUNT_ID}.dkr.ecr.${params.AWS_REGION}.amazonaws.com

              sudo docker pull ${IMAGE}
              sudo docker rm -f ${params.CONTAINER_NAME} || true

              # Run new container (publish HOST_PORT -> CONTAINER_PORT)
              sudo docker run -d --name ${params.CONTAINER_NAME} --restart unless-stopped \
                -e MODEL_PATH=/app/${params.MODEL_FILE} \
                -p ${params.HOST_PORT}:${params.CONTAINER_PORT} \
                ${IMAGE}

              # Quick health probe on host port (mapped to container)
              for i in {1..10}; do
                sleep 3
                if curl -fsS http://127.0.0.1:${params.HOST_PORT}/health >/dev/null; then
                  echo "Health OK"
                  exit 0
                fi
              done
              echo "Health check failed" >&2
              sudo docker logs ${params.CONTAINER_NAME} || true
              exit 1
            '
          """
        }
      }
    }
  }

  post {
    success {
      echo "✅ Deployed ${IMAGE} to ${params.EC2_HOST} as ${params.CONTAINER_NAME}"
    }
    failure {
      echo "❌ Pipeline failed. Check logs above."
    }
    always {
      sh 'docker image prune -f || true'
    }
  }
}
