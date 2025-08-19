pipeline {
  agent any
  environment {
    APP_NAME = "mobile-price-app"
    IMAGE_TAG = "latest"
  }
  stages {
    stage('Checkout') { steps { checkout scm } }

    stage('Build image') {
      steps {
        sh 'docker version'
        sh 'docker build -t ${APP_NAME}:${IMAGE_TAG} .'
      }
    }

    stage('Deploy') {
      steps {
        sh '''
          docker rm -f ${APP_NAME} || true
          docker run -d --name ${APP_NAME} -p 8000:8000 ${APP_NAME}:${IMAGE_TAG}
          sleep 3
          curl -sf http://localhost:8000/health || (docker logs ${APP_NAME} && exit 1)
        '''
      }
    }
  }
  post {
    always { sh 'docker ps -a || true' }
    success { echo "Deployed. Test: http://<EC2-PUBLIC-IP>:8000/health" }
  }
}
