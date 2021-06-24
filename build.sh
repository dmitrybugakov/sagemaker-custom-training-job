#!/bin/bash

image="boosting"
chmod +x code/train
chmod +x code/serve
docker build -t ${image} .

account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
region=${region:-us-east-1}
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

docker build -t ${image} .
docker tag ${image} ${fullname}

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [[ $? -ne 0 ]]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

$(aws ecr get-login --region ${region} --no-include-email)

docker push ${fullname}
