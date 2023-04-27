# FishNet

주의!  
틀릴 가능성이 있습니다.

</br>

## 1. 개요

**주제** : FishNet 구현  
**개요** : CIFAR10 데이터셋에 맞춰 FishNet을 변형하여 image classification 수행 


### 개발환경

```markdown
- IDE : VSCode
- Language : Python
- Library : torch, torchvision, numpy, scikit-learn, tensorboardX 
- Server : GeForce RTX 3090 1개, Ubuntu 20.04
```

### 디렉토리, 파일 구조

```markdown
📦FishNet
┣ 📂models
┃ ┣ 📜block.py
┃ ┣ 📜fish.py
┃ ┣ 📜fishnet.py
┣ 📜dataset.py
┣ 📜evaluate.py
┣ 📜train.py
┣ 📜trainer.py
┣ 📜util.py
┣ 📜.gitignore
┣ 📜README.md
┗ 📜requirements.txt
```
</br>

## 2. 구조  
 
### **FishNet architecture**
<p align="center">
  <img width="600" alt="fig1" src="https://user-images.githubusercontent.com/68064510/234520380-89aabb49-ff33-4451-a7c2-950495ac5a89.png">
</p>

</br>

### **FishNet module architecture**

<p align="center">
  <img width="600" alt="fig2" src="https://user-images.githubusercontent.com/68064510/225210897-6a05d548-f9f6-4a02-84ce-f296d9715546.png">
</p>

### **- fishnet.py**

FishNet architecture 구현

### **- fish.py**

block.py에서 구현된 block을 이용하여 FishNet의 fish tail, fish body, fish head 생성


### **- blocks.py**

architecture를 구현하기 위해 사용되는 block

- **BRU(Bottleneck Residual Unit)**
    
    Isolated convolution인 기존의 residual block을 대신하여 FishNet에서 사용되는 Bottleneck Residual Block
    
- **UR-block(Up-sampling & Refinement block)**
    
    Fish body에서 사용되는 block 
    
    FishNet module architecture (b) 이미지 구현
    

- **DR_Block(Down-sampling & Refinement block)**
    
    Fish head에서 사용되는 block 
    
    FishNet module architecture (c) 이미지 구현
    
- **SE_Block(Squeeze and Excitation block)**
    
    Fish tail과 Fish body를 연결하는 bridge 

</br>    

## 3. 코드 실행

### 데이터 다운로드

```bash
bash cifar10.sh
```

### 라이브러리 다운로드

```bash
pip install -r requirements.txt
```

### training

```bash
python3 train.py
```

### evaluation

```bash
python3 evaluate.py
```

</br>

## 4. 실험 결과

### - Accuracy 비교 결과
| <center></center> |  <center>Train accuracy</center> |  <center>Validate accuracy</center> |  <center>Test accuracy</center> |
|:--------:|:--------|:--------|:--------|
|**결과** |99.99% |88.9%|88.84% |


### - 기존 모델과 다른점

- data augmentation을 통해 224X224로 input을 resize하지 않고 cifar10 input size 32X32 그대로 실험 진행
    
- Dropout을 적용한 Fully Connected layer 추가  

- concat 되는 시점이 다름
    
</br>

## 5. Reference

### Paper

- [FishNet](https://arxiv.org/abs/1901.03495)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

### Github repository

- [FishNet](https://github.com/kevin-ssy/FishNet)
