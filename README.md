<!-- PROJECT LOGO -->
<br />
<p align="center">
<h3 align="center">DeepChessAI</h3>

  <p align="center">
    BINUS International Intelligence System Final Project
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
   <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#self-play">Self Play</a></li>
         <li><a href="#supervised-learning">Supervised Learning</a></li>
         <li><a href="#evaluate-only">Evaluate only</a></li>
      </ul>
    </li>
  </ol>
</details>

<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/fauzanardh/DeepChessAI.git
   ```
2. Install the Dependencies
   ```sh
   pip install -r requirements.txt
   ```



<!-- USAGE EXAMPLES -->
## Usage
### Self Play
First, you need to generate a new config file
```python
from config import Config
Config().save_config("config-default.json")
```
After creating the config file, run the file `run_do_all.py`
```shell
python run_do_all.py
```

### Supervised Learning
First, you will have to download PGN files and put them in the directory you set in the config file.
You can download the PGN files from [FICS](http://ficsgames.org/download.html).

Second, export the data to TFRecord files by using the `run_exporter.py`
```shell
python run_exporter.py
```

After you export the PGN files, you just need to run the `run_optimize.py` (Train the model)
```shell
python run_optimize.py
```

### Evaluate only
If you only want to evaluate which one is the better model run the `run_evaluate.py`
```shell
python run_evaluate.py
```