## File structure
```
pytorch-template/
│
├── init_project.py - initialize project with config
├── train.py - main script to start training
├── inference.py - inference of trained model
│
├── config/ - abstract base classes
│   └── config.json - configuration for training
│
├── custom_dataset/ - anything about data processing
│   ├── dataset.py - custom dataset
│   └── transformer.py - get transforms with config
│
├── data/ - default directory for storing input data
│
├── model/ - models, losses, and metrics
│   ├── model.py
│   ├── metrics.py
│   └── loss.py
│
├── saved/
│   ├── models/ - trained models are saved here
│   └── log/ - default logdir for logging output
│
├── trainer/ - trainers
│   └── trainer.py
│
└── trainer/ - trainers
```