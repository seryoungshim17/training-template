from init_project import JsonConfig

if __name__ == '__main__':
    cfg = JsonConfig(file_path='./config/__base__.json')
    print(cfg)