# Extract scenes from movies

You probably don't want to fork it :stuck_out_tongue_winking_eye: Steps:

1) Scrap some video
2) Extract scene
3) Run

## 1) Scrap

Scrap videos using `youtube-dl` or `instaloader`. You can use `tools/scrapping.py` to _automate_ this step.

```bash
python tools/scrapping.py --source youtube --count 100
```

See [tools/scrapping.py](tools/scrapping.py) for details.

## 2) Extract scenes from video

```bash
python featurescoop/extract_scenes.py --name mydataset --input inputfolder --output outputfolder --source youtube 
```

See [featurescoop/extract_scenes.py](featurescoop/extract_scenes.py) for details. The settings are stored in `featurescoop/settings/mydataset.json`

## 3) Run API

### 3.1) Using command line

```bash
python featurescoop/api.py
```

### 3.2) Using Docker (Windows)

#### 3.2.1) Build

```bash
.\docker_build.bat
```

#### 3.2.2) Run

Batch script

```bash
.\docker_run.bat arg1 arg2
```

for example:

```bash
.\docker_run.bat -d --restart=unless-stopped
```

Or

```bash
.\docker_run.bat --rm
```

## 4) API

`http://127.0.0.1:5003/settings`  
Name of all settings that can be used.

`http://127.0.0.1:5003/getscenes?name=setting_name&imgs='/path/to/img1.jpg','/path/to/img2.jpg'`  
Get scenes for the settings named _setting_name_
