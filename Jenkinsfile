pipeline {
  agent any
  options { timestamps(); ansiColor('xterm') }

  parameters {
    string(name:'AWS_REGION',  defaultValue:'eu-west-2', description:'AWS region')
    string(name:'ACCOUNT_ID',  defaultValue:'430006376054', description:'AWS account id')
    string(name:'ECR_REPO',    defaultValue:'mobile-price-pred', description:'ECR repo name')
    string(name:'EC2_HOST',    defaultValue:'18.175.251.68', description:'EC2 public IP/DNS')
    string(name:'EC2_USER',    defaultValue:'ubuntu', description:'ubuntu or ec2-user')
    string(name:'APP_PORT',    defaultValue:'8080', description:'container exposes this port')
    string(name:'CONTAINER_NAME', defaultValue:'mobile-price-app')
  }

  environment {
    ECR_URI   = "${params.ACCOUNT_ID}.dkr.ecr.${params.AWS_REGION}.amazonaws.com/${params.ECR_REPO}"
    IMAGE_TAG = "build-${env.BUILD_NUMBER}"
  }

  stages {
    stage('Checkout') { steps { checkout scm } }

    stage('Build image') {
      steps { sh "docker build -t ${env.ECR_URI}:${env.IMAGE_TAG} ." }
    }

    stage('Push to ECR') {
      steps {
        withCredentials([[$class:'AmazonWebServicesCredentialsBinding', credentialsId:'aws-creds']]) {
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
          aws ecr get-login-password --region ${params.AWS_REGION} \
            | docker login --username AWS --password-stdin ${params.ACCOUNT_ID}.dkr.ecr.${params.AWS_REGION}.amazonaws.com

          docker pull ${env.ECR_URI}:latest
          docker rm -f ${params.CONTAINER_NAME} || true

          docker run -d --name ${params.CONTAINER_NAME} --restart unless-stopped \
            -p ${params.APP_PORT}:${params.APP_PORT} \
            --env-file /opt/mobile-price/app.env \
            ${env.ECR_URI}:latest

          sleep 3
          curl -sf http://localhost:${params.APP_PORT}/health || true
        '
      """
    }
  }
}  }
}
