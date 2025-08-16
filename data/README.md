## MVTec-2K

Google Drive: [MVTec-2K.zip](https://drive.google.com/file/d/1giNfM75RWnciIH9KJUIygU-6_aWikoBh/view?usp=drive_link)  
Hugging Face: [XimiaoZhang/MVTec-2K](https://huggingface.co/datasets/XimiaoZhang/MVTec-2K)  
```
huggingface-cli download --repo-type dataset  XimiaoZhang/MVTec-2K --local-dir MVTec-2K --resume-download
```
  
## VisA-2K
  
Google Drive: [VisA-2K.zip](https://drive.google.com/file/d/1kg6rhVPT-zwsleSZi_-6Hlu9D6TxS3ut/view?usp=drive_link)  
Hugging Face: [XimiaoZhang/VisA-2K](https://huggingface.co/datasets/XimiaoZhang/VisA-2K)  
```
huggingface-cli download --repo-type dataset  XimiaoZhang/VisA-2K --local-dir VisA-2K --resume-download
```
  
## MVTec-4K

Google Drive: [MVTec-4K.zip](https://drive.google.com/file/d/10cY3sel_bqlPrqfPCv-yGVQPU2rSe7nQ/view?usp=drive_link)  
Hugging Face: [XimiaoZhang/MVTec-4K](https://huggingface.co/datasets/XimiaoZhang/MVTec-4K)  
```
huggingface-cli download --repo-type dataset  XimiaoZhang/MVTec-4K --local-dir MVTec-4K --resume-download
```
  
## RealIAD-2K
[First apply for the Real-IAD dataset.](https://huggingface.co/datasets/Real-IAD/Real-IAD)  
  
Download the Real-IAD dataset using the following command:  
```
huggingface-cli login    # Login your account
huggingface-cli download --repo-type dataset Real-IAD/Real-IAD realiad_raw/bottle_cap.zip --local-dir ./ --resume-download
huggingface-cli download --repo-type dataset Real-IAD/Real-IAD realiad_raw/mint.zip --local-dir ./ --resume-download
huggingface-cli download --repo-type dataset Real-IAD/Real-IAD realiad_raw/usb_adaptor.zip --local-dir ./ --resume-download
```
`Unzip` them and get the following directory structure:  
```
    |--data                         
        |--RealIAD-2K            
            |--bottle_cap
              |--test.jsonl
              |--train.jsonl
            |--...
            |--test_uni.jsonl   
            |--train_uni.jsonl   
        |--realiad_raw
            |--bottle_cap
              |--NG
              |--OK
            |--...
        |--create_realiad2k.py
```  
Run `python create_realiad2k.py` to create the dataset.  

   
### Please place the downloaded datasets in this folder.   
  
If you are using `anomaly synthesis-based methods` such as DeSTSeg, RealNet, or RD++, you will also need to download the [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) dataset.

### The complete directory structure is as follows:
```
    |--data                         
        |--MVTec-2K            
            |--bottle
            |--capsule
            |--...
            |--test_uni.jsonl   
            |--train_uni.jsonl   
        |--RealIAD-2K
            |--bottle_cap
            |--mint
            |--...
            |--test_uni.jsonl   
            |--train_uni.jsonl      
        |--...    
        |--DTD          
            |--images
            |--labels
            |--imdb 

```

