repo_dir=$(pwd)
# data_dir="${repo_dir}/data"
docker rm tensorflow || true
docker run -it \
    --name tensorflow \
    -p 8888:8888 \
    --mount type=bind,source=$repo_dir,target=/tf \
    --gpus all \
    imagevae 
