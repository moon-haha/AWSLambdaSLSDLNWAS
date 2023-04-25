# AWS Lambda에서 사용할 공식 Node.js 런타임 이미지를 가져옵니다.
FROM public.ecr.aws/lambda/nodejs:16

# Install required packages for building native addons
RUN yum -y update && yum -y install gcc-c++ make python3

# 애플리케이션의 의존성을 설치합니다.
COPY package*.json ./

# AWS Lambda 환경에 맞춰 TensorFlow 패키지를 설치합니다.
RUN npm install

# 애플리케이션 코드를 복사합니다.
COPY . .

ENTRYPOINT ["/lambda-entrypoint.sh"]
# AWS Lambda 함수의 시작점을 정의합니다.
CMD ["app.handler"]
