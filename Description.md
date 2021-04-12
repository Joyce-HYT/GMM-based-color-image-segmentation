# GMM-based Color Image Segmentation
###### tags: `Course`

## 作業要求
撰寫Gaussian Mixture Model，辨識兩張足球比賽畫面及其對應的mask (判別每一個pixel是屬於場地或非場地)

## 程式執行環境
作業系統：Windows 10
程式語言：Python 3.8
IDE：PyCharm 2017.3.2

## 程式說明
1. 用cv2來讀圖檔及轉換顏色，本次作業我使用了RGB和GRAY兩種顏色來建構GMM
```python
img1 = cv2.imread("soccer1.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
``` 

2. GaussianMixture Model的input為2維陣列，因此需要對圖片reshape (高*寬、顏色維度) 
```python
img1 = img1.reshape(225*400, 1)
``` 

3. 本次model僅需判斷場地及非場地兩個元素，因此n_components設為2
```python
M1 = GaussianMixture(n_components=2).fit(img1)
sce1 = M1.predict(img1)
``` 

4. 將預測結果存成csv檔，並與助教給的mask.csv進行比對，計算pixel accuracy
```python
result1 = np.asarray(sce1)
np.savetxt('sec1_RGBresult.csv', result1, delimiter=',')
```  

5. 預測結果需reshape回原本的寬*高，才能印出原本大小的圖片
```python
sce1 = sce1.reshape(225, 400) # 再reshape回來
plt.imshow(sce1)
plt.show()
```

6. 在建構M2時，使用np.array把兩張圖片存起來，再一起丟進model訓練
```python
mix_2_img = np.array(img1)
mix_2_img = np.append(mix_2_img, img2, 0)
M2 = GaussianMixture(n_components=2).fit(mix_2_img)
```  

## Scenarios
### Scenario 1: 以soccer1.jpg建構GMM (=M1)，並以soccer1.jpg做測試
 	
| 圖片         | ![](https://i.imgur.com/RjP8CO5.png) | ![](https://i.imgur.com/MhkjLTf.png) |
|-| -------- | -------- |
|**準確率**| 0.971 | 0.94     |
|**測試模型**| M1 (RGB) | M1 (GRAY) |

以同樣的圖片建構及測試同一個模型，就可以得出很好的效果，然而我使用灰階圖片來建構模型所預測出來的結果，我認為比RGB的效果還要來得好，可以從M1 (GRAY) 的圖片看出，其場地與非場地的分類較為明確。
不過該表格中的準確度是與助教提供的mask.csv來計算的，因此準確度較低 ( M1 (RGB)分類的結果與助教標註的圖片較為相近 )

### Scenario 2: 以M1對soccer2.jpg測試

|圖片| ![](https://i.imgur.com/InYPRSH.png) | ![](https://i.imgur.com/0qx3S0K.png) |
|-| -------- | -------- |
|**準確率**| 0.422 | 0.326     |
|**測試模型**| M1 (RGB) | M1 (GRAY) |

拿不同圖片丟入M1(RGB)預測，其效果非常得差，原因可能是訓練資料不足，或是soccer1和soccer2兩張圖片判定為場地/非場地的位置有差異，因此拿soccer1訓練出來的M1，去判斷soccer2會有很大的誤差。
而我使用灰階圖片所訓練出來的M1，直接看圖片M1(GRAY)，其分類效果是很好的，但因為該預測結果在場地/非場地的判斷上，有混淆的情況，譬如一般定義場地為「1」，而此模型判定場地為「0」，所以直接用mask.csv計算準確度反而更差。

### Scenario 3: 以soccer1 & 2.jpg 建構GMM (=M2)，並各以兩張圖做測試
* **測試模型 M2 (RGB)**

|圖片| ![](https://i.imgur.com/YIJXhLn.png)|![](https://i.imgur.com/9DrLVXL.png)|
|-| -------- | -------- |
|**準確率**| 0.975 | 0.915 |

用兩張圖建構的M2，再分別去分類兩張圖片的場地元素，可以得到很好的效果，但因為是用同樣的圖片去fit並使用同樣的圖片去predict，效果本來就不會多差。
相較於M1，兩張圖片分類的效果都更好了，其中soccer1.jpg分類的更精確，可能是因為訓練資料變多的關係。

* **測試模型 M2 (GRAY)**

|圖片| ![](https://i.imgur.com/pGPbuzE.png) | ![](https://i.imgur.com/U3tHer4.png)|
|-| -------- | -------- |
|**準確率**| 0.909 | 0.695 |

使用灰階圖片建構的M2，它雖然在判斷場上的人物跟罰球線不是很精確，但是在判斷加油區、看板文字甚至節目名稱上，分類的非常精細，如紅框處，M2將看板上的COM也辨識出來。

## 效能 (Pixel accuracy)


| 模型 | M1 (RGB) |M1 (GRAY)| M2 (RGB) | M2 (GRAY) |
| -------- | -------- | --- | --- | -------- |
| soccer1.jpg | 0.971 | 0.94 |0.975 |0.91 |
| soccer2.jpg |0.422 |0.326 |0.915 | 0.695 |

以RGB建構的模型中，M2優於M1。

GRAY與RGB模型相比，雖然依照mask.csv所計算出來的準確度，RGB都是比較好的，但是直接觀察圖片，GRAY分類出來的結果比較明顯且精確。我覺得原因是RGB用3個維度來表示圖片，但GRAY圖片只以0和1來表示，因此在圖片成份比較單純的情況下，model比較容易讀懂圖片，能更精確的分析場地與非場地元素。
