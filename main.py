pip install -U scikit-learn
import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.write("""
# تشخیص شدت اختلال افسردگی به کمک
# هوش مصنوعی و یادگیری ماشین با دقت 95درصد
این برنامه برای تشخیص شدت اختلال افسردگی بر اساس اطلاعات ورودی کاربر توسط *کیارمین غفاری* طراحی شده است

""")
st.sidebar.header('ورودی کاربر')

def user_input_features():
    Gender = st.sidebar.radio("جنسیت",['مرد','زن'])
    level = st.sidebar.radio('سطح تحصیلات', ['فوق لیسانس','لیسانس','دیپلم','سیکل','بی سواد'])
    job = st.sidebar.radio('وضعیت شغلی', ['بیکار','آزاد','معلم','کارگر','کارمند','خانه دار'])
    Age = st.sidebar.slider('سن', 1, 100, 1)
    marrid = st.sidebar.radio('وضعیت تاهل',['مجرد','متاهل','طلاق گرفته','همسر فوت کرده'])
    child = st.sidebar.slider('تعداد فرزند',0,10,0)
    sport = st.sidebar.radio('میزان ورزش کردن',['حرفه ای','گاهی اوقات','کم'])
    music = st.sidebar.radio('میزان گوش دادن به موسیقی',['زیاد','متوسط','کم'])
    faith = st.sidebar.radio('میزان اعتقادات دینی و مذهبی',['زیاد','متوسط','کم'])
    travel = st.sidebar.radio('میزان سفر کردن',['زیاد','متوسط','کم'])
    economic = st.sidebar.radio('وضعیت مالی',['زیاد','متوسط','کم'])
    data = {'Gender':Gender,
            'level':level,
            'job':job,
            'Age':Age,
            'marrid':marrid,
            'child':child,
            'sport':sport,
            'music':music,
            'faith':faith,
            'travel':travel,
            'economic':economic}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

#st.subheader('ورودی کاربر')
df['Gender'] = df['Gender'].map({'مرد':1,'زن':0})
df['level']=df['level'].map({'بی سواد':0,'سیکل':1,'دیپلم':2,'لیسانس':3,'فوق لیسانس':4})
df['job']=df['job'].map({'خانه دار':0,'کارمند':1,'معلم':2,'آزاد':3,'بیکار':4,'کارگر':5})
df['marrid']=df['marrid'].map({'متاهل':0,'مجرد':1,'طلاق گرفته':2,'همسر فوت کرده':3})
df['sport']=df['sport'].map({'کم':0,'گاهی اوقات':1,'حرفه ای':2})
df['music']=df['music'].map({'کم':0,'متوسط':1,'زیاد':2})
df['faith']=df['faith'].map({'کم':0,'متوسط':1,'زیاد':2})
df['travel']=df['travel'].map({'کم':0,'متوسط':1,'زیاد':2})
df['economic']=df['economic'].map({'کم':0,'متوسط':1,'زیاد':2})

#st.write(df)

ds = pd.read_excel(r"C:\Users\DEAR USER\Desktop\208331495453976.xls")


ds['1.Gender'] = ds['1.Gender'].map({'M':1,'F':0})
ds['2.level']=ds['2.level'].map({'Unlettered':0,'cycle':1,'Diplom':2,'Bachelor':3,'Master':4})
ds['3. job']=ds['3. job'].map({'homey':0,'employe':1,'teacher':2,'free':3,'Idle':4,'worker':5})
ds['5. marrid']=ds['5. marrid'].map({'married':0,'single':1,'divorce':2,'widow':3})
ds['7.sport']=ds['7.sport'].map({'low':0,'Sometimes':1,'profesional':2})
ds['8.music']=ds['8.music'].map({'low':0,'medium':1,'much':2})
ds['9.Faith']=ds['9.Faith'].map({'low':0,'medium':1,'high':2})
ds['10.Travel']=ds['10.Travel'].map({'low':0,'medium':1,'many':2})
ds['11.mali']=ds['11.mali'].map({'low':0,'medium':1,'high':2})
ds['14.Dejection']=ds['14.Dejection'].map({'low':0,'Medium':1,'Much':2})
ds = ds.dropna()
#st.write(ds)
X=ds.drop(columns=['14.Dejection'])
y=ds[['14.Dejection']]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rfc = RandomForestClassifier(criterion = 'entropy', random_state = 42)
rfc.fit(x_train, y_train)

prediction = rfc.predict(df)


st.subheader('(کم ,متوسط,زیاد):نتیجه')


if prediction == 0:
    st.title('کم')
    st.write('تبریک میگم شما در وضعیت خوبی هستید ')
    from PIL import Image
    photo = Image.open("InShot_۲۰۲۳۰۶۲۶_۱۴۰۴۳۰۶۲۹.jpg")
    st.image(photo)
    st.balloons()
elif prediction == 1:
    st.title('متوسط')
    from PIL import Image
    photo = Image.open("InShot_۲۰۲۳۰۶۲۶_۱۴۰۹۴۴۷۳۴.jpg")
    st.image(photo)
elif prediction == 2:
    st.title('زیاد')
    from PIL import Image
    photo = Image.open("InShot_۲۰۲۳۰۶۲۶_۱۴۱۳۳۱۴۴۶.jpg")
    st.image(photo)
    













