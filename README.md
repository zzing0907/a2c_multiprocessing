# A2C Multiprocessing    

Advantage Actor-Critic with Multiprocessing   

Multiprocessing EnvWrapper from [OpenAI baselines code](https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py)  

<br>

### Usage  

```
## default settings  
python3 a2c_multi.py   
  
## setting environment and workers(number of process)  
python3 a2c_multi.py --env_name Boxing --n_workers 32  
```

 