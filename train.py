from init_project import json_config

if __name__ == '__main__':
    cfg = json_config(file_path='./config/__base__.json')
    print(cfg)