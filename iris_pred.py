import streamlit as st
import joblib

# 載入封裝好的model
svm_clf = joblib.load("svm_clf_model.joblib")
knn_clf = joblib.load("knn_clf_model.joblib")
rf_clf = joblib.load("rf_clf_model.joblib")

st.title("⚜️鳶尾花品種預測⚜️ 龐宇宸")

clf = st.sidebar.selectbox("### 請選擇分類模型:",
                           ["KNN","SVM","RandomForest"])

s1 = st.slider("花萼長度:",3.0,8.0,5.8) # 5.8預設的位置
s2 = st.slider("花萼寬度:",2.0,5.0,3.5)
s3 = st.slider("花瓣長度:",1.0,7.0,4.5)
s4 = st.slider("花瓣寬度:",0.1,2.6,1.2)

labels = ['Setosa','Versicolor','Virginica']

if clf == "KNN":
    clf_model = knn_clf
elif clf == "SVM":
    clf_model = svm_clf
else:
    clf_model = rf_clf

if st.button("進行預測"):
    X = [[s1,s2,s3,s4]]
    y = clf_model.predict(X) # 預測結果是一維(情況可能是n列x1行，因此要取第一個值出來)
    #st.write(y[0])
    st.write("#### 預測品種:",labels[y[0]])
    if labels[y[0]] == 'Setosa':
        st.image('setosa.jpg',width = 300)
        st.info("山鳶尾: 這種鳶尾花的特徵包括小而短的花瓣和小花萼。花瓣通常是藍色、紫色或白色，且通常具有鮮豔的顏色。這種鳶尾生長在寒冷和潮濕的環境中。")
    elif labels[y[0]] == 'Versicolor':
        st.image('versicolor.jpg',width = 300)
        st.info("變色鳶尾: 這種鳶尾花的特徵包括中等大小的花瓣和花萼。花瓣可能是藍色、紫色、粉紅色或白色。這種鳶尾生長在相對潮濕的地區，比如沼澤和溪流旁邊。")
    elif labels[y[0]] == 'Virginica':
        st.image('virginica.jpg',width = 300)
        st.info("維吉尼亞鳶尾: 這種鳶尾花的特徵包括大而長的花瓣和花萼。花瓣可能是紫色、白色或粉紅色。這種鳶尾生長在潮濕且較為溫暖的環境中。")
# 因為 y 是 Numpy 陣列（array），而不是單個的整數值。
# 直接將整個 Numpy 陣列傳遞給 labels 列表將無法正確索引到對應的品種名稱。
