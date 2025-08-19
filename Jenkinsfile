
pipeline {
  agent any
  options { timestamps(); ansiColor('xterm') }

  parameters {
    string(name: 'AWS_REGION',       defaultValue: 'eu-west-2')
    string(name: 'ACCOUNT_ID',       defaultValue: '430006376054')
    string(name: 'ECR_REPO',         defaultValue: 'mobile_price_pred_app')
    string(name: 'EC2_HOST',         defaultValue: '18.175.251.68')
    string(name: 'EC2_USER',         defaultValue: 'ubuntu')
    string(name: 'HOST_PORT',        defaultValue: '8000', description: 'Public port on EC2')
    string(name: 'CONTAINER_PORT',   defaultValue: '8000', description: 'Uvicorn port in container')
    string(name: 'CONTAINER_NAME',   defaultValue: 'mobile-price-app')
  }

  environment {
    ECR_URI   = "${params.ACCOUNT_ID}.dkr.ecr.${params.AWS_REGION}.amazonaws.com/${params.ECR_REPO}"
    IMAGE_TAG = "build-${env.BUILD_NUMBER}"
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Build image') {
      steps {
        sh "docker build -t ${env.ECR_URI}:${env.IMAGE_TAG} ."
      }
    }

    stage('Push to ECR') {
      steps {
        withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-creds']]) {
          sh """
            aws ecr get-login-password --region ${params.AWS_REGION} \
              | docker login --username AWS --password-stdin ${params.ACCOUNT_ID}.dkr.ecr.${params.AWS_REGION}.amazonaws.com

            aws ecr describe-repositories --repository-names ${params.ECR_REPO} --region ${params.AWS_REGION} \
              || aws ecr create-repository --repository-name ${params.ECR_REPO} --region ${params.AWS_REGION}

            docker push ${env.ECR_URI}:${env.IMAGE_TAG}
            docker tag ${env.ECR_URI}:${env.IMAGE_TAG} ${env.ECR_URI}:latest
            docker push ${env.ECR_URI}:latest
          """
        }
      }
    }

    stage('Deploy to EC2') {
      steps {
        sshagent(credentials: ['ec2-ssh-key']) {
          sh """
            ssh -o StrictHostKeyChecking=no ${params.EC2_USER}@${params.EC2_HOST} '
              set -e
              aws ecr get-login-password --region ${params.AWS_REGION} |
                docker login --username AWS --password-stdin ${params.ACCOUNT_ID}.dkr.ecr.${params.AWS_REGION}.amazonaws.com

              docker pull ${env.ECR_URI}:latest
              docker rm -f ${params.CONTAINER_NAME} || true

              docker run -d --name ${params.CONTAINER_NAME} --restart unless-stopped \
                -p ${params.HOST_PORT}:${params.CONTAINER_PORT} \
                --env-file /opt/mobile-price/app.env \
                ${env.ECR_URI}:latest

              sleep 3
              curl -sf http://localhost:${params.HOST_PORT}/health || true
            '
          """
        }
      }
    }
  }
}
