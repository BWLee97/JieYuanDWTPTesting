import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
print(tf.__version__)
print(kt.__version__)
import matplotlib
print(matplotlib.__version__)
# 将页面放大至适应web宽度
st.set_page_config(layout="wide")
# 定义检查Excel文件的函数以便于后续调用
def check_sheet_exists(file_path, sheet_name):
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        if sheet_name in sheet_names:
            return True
        else:
            return False
    except:
        return False
# 定义数据形式转换函数以便于后续深度学习模型的训练
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)
# 第一部分的第一页
def page_11():
    st.header('项目简介')
    with st.container(border=True):
        st.subheader('Background')
        st.write('111')
# 第一部分的第二页
def page_12():
    st.header('项目简介')
    with st.container(border=True):
        st.subheader('Background')
        st.write('111')
# 第二部分的第一页：数据上传并显示
def page_21():
    # 设置本页标题
    st.header('数据上传')
    # 上传训练数据
    with st.container(border=True):
        # 设置子标题
        st.subheader('上传训练数据')
        # 设置上传模块
        uploaded_file = st.file_uploader("在上传数据前请阅读**注意事项**，并根据要求上传。")
    # 显示上传的数据
    with st.container(border=True):
        # 设置子标题
        st.subheader('加载数据')
        # 检查是否上传了数据
        if uploaded_file is not None:
            with st.spinner('请耐心等待...'):
                # 转换为dataframe格式准备储存为excel以供后续调用
                dataframe = pd.read_excel(uploaded_file)
                # 转换时间格式
                dataframe['时间'] = pd.to_datetime(dataframe['时间'],
                                                 format='%Y-%m-%d %H:%M:%S')
                # 将时间信息转换为索引
                dataframe.index = dataframe['时间']
                # 删除多余的时间列
                dataframe.drop(columns=['时间'], axis=1, inplace=True)
                # 检查是否存在缓存文件夹，若没有则创建缓存文件夹
                if not os.path.exists('Cache'):
                    os.mkdir('Cache')
                if not os.path.exists('Cache/model'):
                    os.mkdir('Cache/model')
                if not os.path.exists('Cache/history'):
                    os.mkdir('Cache/history')
                # 将上传的数据保存于缓存目录
                dataframe.to_excel('Cache/raw data.xlsx', sheet_name='raw')
                # 成功上传提示
                st.success('上传成功！', icon="✅")
                # 显示已经上传的数据
                st.caption('你上传的数据如下所示：')
                st.write(dataframe)
        # 提示上传数据
        else:
            st.info('请上传训练数据！', icon="ℹ️")
# 第二部分的第二页：缺失值填充、数据归一化
def page_22():
    # 设置本页标题
    st.header('数据处理')
    with st.container(border=True):
        # 设置子标题
        st.subheader('填充缺失值')
        # 检查是否上传了数据，若没有则提醒
        with st.spinner('检测数据中，请稍后...'):
            if check_sheet_exists(file_path='Cache/raw data.xlsx',
                                  sheet_name='raw'):
                # 从缓存目录中读取数据
                data = pd.read_excel('Cache/raw data.xlsx', sheet_name='raw')
                # 转换时间格式
                data['时间'] = pd.to_datetime(data['时间'],
                                            format='%Y-%m-%d %H:%M:%S')
                # 提取时间信息并将时间信息转换为索引
                date_index = data['时间']
                data.index = date_index
                # 删除多余的时间列
                data.drop(columns=['时间'], axis=1, inplace=True)
                # 提示成功检测
                st.success("已检测到上传数据！", icon="✅")
                # 布局，将询问缺失值是否存在和选择填充方法分成两列
                col_22_1, col_22_2 = st.columns(2)
                # 询问用户是否包含缺失值
                with col_22_1:
                    missing = st.radio("是否包含缺失值？", options=["是", "否"],
                                       index=None)
                # 如果用户确定缺失值存在，则要求用户选择填充方法，否则禁用该选项
                with col_22_2:
                    if missing == "是":
                        missing_option = st.selectbox("请选择缺失值填充方法：",
                                                      ("KNN填充", "移动平均滤波"),
                                                      index=None,
                                                      placeholder="单击此处以选择方法")
                    else:
                        missing_option = st.selectbox("请选择缺失值填充方法：",
                                                      ("KNN填充", "移动平均滤波"),
                                                      index=None,
                                                      placeholder="单击此处以选择方法",
                                                      disabled=True)
                # 如果包含缺失值，则根据选择的填充方法填充缺失值
                if missing == "是":
                    # 使用移动平均滤波填充缺失值
                    if missing_option == "移动平均滤波":
                        # 设置提交表单
                        with st.form('MAF_form'):
                            # 要求用户输入时间窗口值
                            window_size = st.number_input("请输入时间窗口值：",
                                                          value=1)
                            # 设置提交按钮
                            submitted = st.form_submit_button('提交')
                        # 如果用户提交时间窗口值，则利用移动平均滤波方法进行填充
                        if submitted:
                            # 线性填充缺失值
                            def fill_missing_values(data):
                                filled_data = np.copy(data).astype(float)
                                missing_indices = np.isnan(filled_data)
                                filled_data[missing_indices] = np.interp(np.flatnonzero(missing_indices),
                                                                         np.flatnonzero(~missing_indices),
                                                                         filled_data[~missing_indices])
                                return filled_data
                            # 移动平均滤波，增强时序特性
                            def moving_average_smoothing(data, window_size):
                                smoothed_data = np.convolve(data,
                                                            np.ones(window_size)/window_size,
                                                            mode='same')
                                return smoothed_data
                            # 转换为NumPy数组以便后续填充
                            numpy_data = data.to_numpy()
                            # 按列进行线性填充，并在填充结束后进行移动平均滤波以增强时序特性
                            fill_data = []
                            for i in range(numpy_data.shape[1]):
                                x = fill_missing_values(numpy_data[:, i])
                                x = moving_average_smoothing(x, window_size)
                                fill_data.append(x)
                            # 将numpy数组转换回Dataframe
                            fill_data = pd.DataFrame(fill_data)
                            # 对Dataframe进行转置以正确显示数据
                            fill_data = fill_data.T
                            # 将索引转换为时间信息
                            fill_data.index = date_index
                            # 将列名改为正确的列名，与data相同
                            fill_data.columns = data.columns
                            # 将填充好的数据保存于缓存目录
                            fill_data.to_excel('Cache/imputed data.xlsx',
                                               sheet_name='filled')
                            # 选择是否显示填充好的数据
                            with st.expander('您的数据如下所示：'):
                                st.write(fill_data)
                        # 否则提示提交K值
                        else:
                            with st.expander('您的数据如下所示：', expanded=True):
                                st.info('请为移动平均滤波方法选择合适的时间窗口值！', icon="ℹ️")
                    elif missing_option == "KNN填充":
                        # 设置提交表单
                        with st.form('KNN_form'):
                            # 要求用户指定K值
                            n_neighbors = st.number_input("请输入K值：", value=1,
                                                          min_value=1,
                                                          max_value=10)
                            # 设置提交按钮
                            submitted = st.form_submit_button('提交')
                        # 如果用户提交K值，则利用KNN方法进行填充
                        if submitted:
                            # KNN填充
                            imputer = KNNImputer(n_neighbors=n_neighbors)
                            impute_data = imputer.fit_transform(data)
                            # 将数据转换回Dataframe
                            impute_data = pd.DataFrame(impute_data)
                            # 将索引转换为时间信息
                            impute_data.index = date_index
                            # 将列名改为正确的列名，与data相同
                            impute_data.columns = data.columns
                            # 将填充好的数据保存于缓存目录
                            impute_data.to_excel('Cache/imputed data.xlsx',
                                                 sheet_name='filled')
                            # 选择是否显示填充好的数据
                            with st.expander('您的数据如下所示：'):
                                st.write(impute_data)
                        # 否则提示提交K值
                        else:
                            with st.expander('您的数据如下所示：', expanded=True):
                                st.info('请为KNN填充方法选择合适的K值！', icon="ℹ️")
                    # 提示用户做出选择
                    else:
                        st.info('请选择是否包含缺失值并选择填充方法！', icon="ℹ️")
                # 如果用户选择没有缺失值，则直接显示原始数据
                elif missing == "否":
                    # 将数据保存于缓存目录
                    data.to_excel('Cache/imputed data.xlsx',
                                  sheet_name='filled')
                    # 选择是否显示数据
                    with st.expander('您的数据如下所示：'):
                        st.write(data)
            # 提示上传数据
            else:
                st.info('请上传训练数据！', icon="ℹ️")
    with st.container(border=True):
        # 设置子标题
        st.subheader('标准化数据')
        # 检查是否填充了缺失值，若没有则提醒
        with st.spinner('检测数据中，请稍后...'):
            if check_sheet_exists(file_path='Cache/imputed data.xlsx',
                                  sheet_name='filled'):
                # 从缓存目录中读取数据
                data = pd.read_excel('Cache/imputed data.xlsx',
                                     sheet_name='filled')
                # 转换时间格式
                data['时间'] = pd.to_datetime(data['时间'],
                                            format='%Y-%m-%d %H:%M:%S')
                # 提取时间信息并将时间信息转换为索引
                data_index = data['时间']
                data.index = data_index
                # 删除多余的时间列
                data.drop(columns=['时间'], axis=1, inplace=True)
                # 提示成功检测
                st.success("已检测到填充数据！", icon="✅")
                norm_option = st.selectbox("请选择数据标准化方法：",
                                           ("最大最小值归一化", "Z值规范化", "鲁棒缩放"),
                                           index=None,
                                           placeholder="单击此处以选择方法")
                if norm_option == "最大最小值归一化":
                    # 最大最小值归一化
                    scaler = preprocessing.MinMaxScaler()
                    # 储存数据标准化方法
                    with open('Cache/scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    # 计算标准化数据
                    scaled_data = scaler.fit_transform(data)
                    # 将数据转换回Dataframe
                    scaled_data = pd.DataFrame(scaled_data)
                    # 将索引转换为时间信息
                    scaled_data.index = data_index
                    # 将列名改为正确的列名，与data相同
                    scaled_data.columns = data.columns
                    # 将填充好的数据保存于缓存目录
                    scaled_data.to_excel('Cache/normalized data.xlsx',
                                         sheet_name='scaled')
                    # 选择是否显示填充好的数据
                    with st.expander('您的标准化数据如下所示：'):
                        st.write(scaled_data)
                elif norm_option == "Z值规范化":
                    # Z值规范化
                    scaler = preprocessing.StandardScaler()
                    # 储存数据标准化方法
                    with open('Cache/scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    # 计算标准化数据
                    scaled_data = scaler.fit_transform(data)
                    # 将数据转换回Dataframe
                    scaled_data = pd.DataFrame(scaled_data)
                    # 将索引转换为时间信息
                    scaled_data.index = date_index
                    # 将列名改为正确的列名，与data相同
                    scaled_data.columns = data.columns
                    # 将填充好的数据保存于缓存目录
                    scaled_data.to_excel('Cache/normalized data.xlsx',
                                         sheet_name='scaled')
                    # 选择是否显示填充好的数据
                    with st.expander('您的标准化数据如下所示：'):
                        st.write(scaled_data)
                elif norm_option == "鲁棒缩放":
                    # 鲁棒缩放
                    scaler = preprocessing.RobustScaler()
                    # 储存数据标准化方法
                    with open('Cache/scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    # 计算标准化数据
                    scaled_data = scaler.fit_transform(data)
                    # 将数据转换回Dataframe
                    scaled_data = pd.DataFrame(scaled_data)
                    # 将索引转换为时间信息
                    scaled_data.index = data_index
                    # 将列名改为正确的列名，与data相同
                    scaled_data.columns = data.columns
                    # 将填充好的数据保存于缓存目录
                    scaled_data.to_excel('Cache/normalized data.xlsx',
                                         sheet_name='scaled')
                    # 选择是否显示填充好的数据
                    with st.expander('您的标准化数据如下所示：'):
                        st.write(scaled_data)
                else:
                    st.info('请选择一种数据标准化方法！', icon="ℹ️")
            # 提示填充数据
            else:
                st.info('请填充训练数据！', icon="ℹ️")
def page_23():
    # 设置本页标题
    st.header('特征选择')
    with st.container(border=True):
        # 设置子标题
        st.subheader('选择变量')
        # 检查是否标准化了数据值，若没有则提醒
        with st.spinner('检测数据中，请稍后...'):
            if check_sheet_exists(file_path='Cache/normalized data.xlsx',
                                  sheet_name='scaled'):
                # 从缓存目录中读取数据
                data = pd.read_excel('Cache/normalized data.xlsx',
                                     sheet_name='scaled')
                # 转换时间格式
                data['时间'] = pd.to_datetime(data['时间'],
                                            format='%Y-%m-%d %H:%M:%S')
                # 提取时间信息并将时间信息转换为索引
                data_index = data['时间']
                data.index = data_index
                # 删除多余的时间列
                data.drop(columns=['时间'], axis=1, inplace=True)
                data_colums = data.columns
                # 提示成功检测
                st.success("已检测到标准化数据！", icon="✅")
                # 设置提交表单
                with st.form('feature_form'):
                    # 要求用户选择输入变量
                    X_options = st.multiselect("**输入变量**", data_colums,
                                               placeholder='请选择模型的输入变量')
                    # 要求用户选择输出变量
                    y_options = st.multiselect("**输出变量**", data_colums,
                                               placeholder='请选择模型的输出变量')
                    # 设置提交按钮
                    submitted = st.form_submit_button('提交')
            # 提示处理数据
            else:
                with st.form('feature_form'):
                    st.info('请完成数据处理！', icon="ℹ️")
                    submitted = st.form_submit_button('提交', disabled=True)
    with st.container(border=True):
        # 设置子标题
        st.subheader('数据集')
        # 提交变量表单后，判断用户是否提交了空的表单
        if submitted:
            # 判断X_options或y_options是否为空，如果为空则给出提示
            if not X_options or not y_options:
                st.error("请至少选择一个输入或输出变量！", icon="🚨")
            # 如果不为空，则将对应的数据储存在缓存目录并显示在前端
            else:
                with st.spinner('划分变量中，请稍后...'):
                    X_new = data[X_options]
                    y_new = data[y_options]
                    # 布局，将输入和输出数据分成左右两部分显示
                    col_X, col_y = st.columns(2)
                    # 左部分显示输入变量的数据
                    with col_X:
                        # 将输入变量储存在缓存目录中
                        X_new.to_excel('Cache/X data.xlsx', sheet_name='X')
                        # 显示输入变量
                        st.caption('您的输入变量如下所示：')
                        st.write(X_new)
                    # 右部分显示输出变量的数据
                    with col_y:
                        # 将输出变量储存在缓存目录中
                        y_new.to_excel('Cache/y data.xlsx', sheet_name='y')
                        # 显示输出变量
                        st.caption('您的输出变量如下所示：')
                        st.write(y_new)
        # 如果用户未提交表单，则提示
        else:
            st.info('请选择期望的输入和输出变量，并点击**提交**！', icon="ℹ️")
def page_31():
    st.header('数据划分')
    with st.container(border=True):
        st.subheader('选择划分比例')
        # 检查是否指定了输入变量，若没有则提醒
        with st.spinner('检测数据中，请稍后...'):
            # 检查输入变量数据和输出变量数据是否存在
            x_data_exists = check_sheet_exists(file_path='Cache/X data.xlsx',
                                               sheet_name='X')
            y_data_exists = check_sheet_exists(file_path='Cache/y data.xlsx',
                                               sheet_name='y')
            if x_data_exists and y_data_exists:
                # 布局，将数据监测提示分成左右两部分显示
                col_31_X, col_31_y = st.columns(2)
                # 左部分提示输入变量检测成功
                with col_31_X:
                    # 从缓存目录中读取输入数据
                    X = pd.read_excel('Cache/X data.xlsx', sheet_name='X')
                    # 转换时间格式
                    X['时间'] = pd.to_datetime(X['时间'],
                                             format='%Y-%m-%d %H:%M:%S')
                    # 提取时间信息并将时间信息转换为索引
                    X_index = X['时间']
                    X.index = X_index
                    # 删除多余的时间列
                    X.drop(columns=['时间'], axis=1, inplace=True)
                    X_colums = X.columns
                    # 提示成功检测
                    st.success("已检测到输入变量数据！", icon="✅")
                # 右部分显示输出变量检测成功
                with col_31_y:
                    # 从缓存目录中读取输出数据
                    y = pd.read_excel('Cache/y data.xlsx', sheet_name='y')
                    # 转换时间格式
                    y['时间'] = pd.to_datetime(y['时间'],
                                             format='%Y-%m-%d %H:%M:%S')
                    # 提取时间信息并将时间信息转换为索引
                    y_index = y['时间']
                    y.index = y_index
                    # 删除多余的时间列
                    y.drop(columns=['时间'], axis=1, inplace=True)
                    y_colums = y.columns
                    # 提示成功检测
                    st.success("已检测到输出变量数据！", icon="✅")
                # 设置提交表单
                with st.form('feature_form'):
                    # 要求用户选择数据划分比例
                    split_ratio = st.slider("请选择数据划分比例，即训练集的占比（%）：",
                                            min_value=50, max_value=90,
                                            value=80)
                    # 设置提交表单
                    submitted = st.form_submit_button('提交')
            else:
                if not x_data_exists:
                    st.error("请选择合适的输入变量", icon="🚨")
                if not y_data_exists:
                    st.error("请选择合适的输出变量", icon="🚨")
    with st.container(border=True):
        st.subheader('训练—测试数据集')
        # 提交表单后，根据用户提供的比例划分数据集
        if submitted:
            with st.spinner('划分数据中，请稍后...'):
                # 合并输入和输出数据
                data = pd.concat([X, y], axis=1)
                data_colums = data.columns
                # 将数据划分为训练数据和测试数据
                data_train, data_test = train_test_split(data, test_size=1-split_ratio/100)
                # 从训练数据和测试数据中提取输入和输出变量
                y_train = data_train[y_colums]
                X_train = data_train[X_colums]
                y_test = data_test[y_colums]
                X_test = data_test[X_colums]
                # 将所有数据储存在缓存目录中以便后续调用
                y_train.to_excel('Cache/y train data.xlsx', sheet_name='y')
                X_train.to_excel('Cache/X train data.xlsx', sheet_name='X')
                y_test.to_excel('Cache/y test data.xlsx', sheet_name='y')
                X_test.to_excel('Cache/X test data.xlsx', sheet_name='X')
                # 选择是否显示训练数据和测试数据中的输入和输出变量
                with st.expander('您的训练数据如下所示：'):
                    # 布局，将训练数据中的输入和输出变量分成左右两部分显示
                    col_X_train, col_y_train = st.columns(2)
                    # 左部分显示训练数据中的输入变量
                    with col_X_train:
                        st.caption('输入变量：')
                        st.write(X_train)
                    # 右部分显示训练数据中的输出变量
                    with col_y_train:
                       st.caption('输出变量：')
                       st.write(y_train)
                # 选择是否显示训练数据和测试数据中的输入和输出变量
                with st.expander('您的测试数据如下所示：'):
                    # 布局，将测试数据中的输入和输出变量分成左右两部分显示
                    col_X_test, col_y_test = st.columns(2)
                    # 左部分显示测试数据中的输入变量
                    with col_X_test:
                        st.caption('输入变量：')
                        st.write(X_test)
                    # 右部分显示测试数据中的输出变量
                    with col_y_test:
                        st.caption('输入变量：')
                        st.write(y_test)
        # 如果用户未提交表单，则提示
        else:
            with st.expander('您的训练数据如下所示：', expanded=True):
                st.info('请选择期望的数据划分比例，并点击**提交**！', icon="ℹ️")
            with st.expander('您的测试数据如下所示：'):
                st.info('请选择期望的数据划分比例，并点击**提交**！', icon="ℹ️")
def page_32():
    st.header('模型训练')
    with st.container(border=True):
        st.subheader('机器学习模型')
        # 检查是否指定了输入变量，若没有则提醒
        with st.spinner('检测数据中，请稍后...'):
            # 检查训练数据集中的输入变量数据和输出变量数据是否存在
            x_train_exists = check_sheet_exists(file_path='Cache/X train data.xlsx', sheet_name='X')
            X_test_exists = check_sheet_exists(file_path='Cache/X test data.xlsx', sheet_name='X')
            if x_train_exists and X_test_exists:
                # 从缓存目录中读取输入数据
                X_train = pd.read_excel('Cache/X train data.xlsx',
                                        sheet_name='X')
                X_test = pd.read_excel('Cache/X test data.xlsx',
                                       sheet_name='X')
                # 转换时间格式
                X_train['时间'] = pd.to_datetime(X_train['时间'],
                                               format='%Y-%m-%d %H:%M:%S')
                X_test['时间'] = pd.to_datetime(X_test['时间'],
                                              format='%Y-%m-%d %H:%M:%S')
                # 提取时间信息并将时间信息转换为索引
                X_train_index = X_train['时间']
                X_test_index = X_test['时间']
                X_train.index = X_train_index
                X_test.index = X_test_index
                # 删除多余的时间列
                X_train.drop(columns=['时间'], axis=1, inplace=True)
                X_test.drop(columns=['时间'], axis=1, inplace=True)
                # 从缓存目录中读取输出数据
                y_train = pd.read_excel('Cache/y train data.xlsx',
                                        sheet_name='y')
                y_test = pd.read_excel('Cache/y test data.xlsx',
                                       sheet_name='y')
                # 转换时间格式
                y_train['时间'] = pd.to_datetime(y_train['时间'],
                                               format='%Y-%m-%d %H:%M:%S')
                y_test['时间'] = pd.to_datetime(y_test['时间'],
                                              format='%Y-%m-%d %H:%M:%S')
                # 提取时间信息并将时间信息转换为索引
                y_train_index = y_train['时间']
                y_test_index = y_test['时间']
                y_train.index = y_train_index
                y_test.index = y_test_index
                # 删除多余的时间列
                y_train.drop(columns=['时间'], axis=1, inplace=True)
                y_test.drop(columns=['时间'], axis=1, inplace=True)
                y_colums = y_train.columns
                # 提示成功检测
                st.success("已检测到训练数据！您可以选择任一模型进行训练！", icon="✅")
                # 第一行展示四个模型
                col1, col2, col3, col4 = st.columns(4, border=True)
                # 第一行第一列：支持向量机模型
                with col1:
                    st.subheader('*支持向量机*')
                    # 点击展示调整超参数界面
                    with st.popover("调整超参数（**不推荐**）",
                                    use_container_width=True):
                        hp11 = st.selectbox("内核类型",
                                            ("linear", "poly", "rbf", "sigmoid", "precomputed"),
                                            index=2)
                        hp21 = st.selectbox("核系数", ("scale", "auto"))
                    # 分为两个按钮，左侧点击进行训练，右侧为超链接跳转
                    col11, col21 = st.columns(2)
                    # 训练模型
                    with col11:
                        if st.button("训练", type="primary",
                                     use_container_width=True):
                            svr = MultiOutputRegressor(sklearn.svm.SVR(kernel=hp11, gamma=hp21))
                            svr = svr.fit(X_train, y_train)
                            # 储存模型
                            with open('Cache/model/svr.pkl', 'wb') as f:
                                pickle.dump(svr, f)
                        if os.path.exists('Cache/model/svr.pkl'):
                            with open('Cache/model/svr.pkl', 'rb') as f:
                                svr = pickle.load(f)
                            # 利用训练好的模型进行预测
                            y_train_pred = svr.predict(X_train)
                            y_test_pred = svr.predict(X_test)
                            # 模型性能评估：训练集中的R2
                            r2_train = []
                            for i in range(y_test.shape[1]):
                                r2_train_i = metrics.r2_score(y_train.iloc[:, i], y_train_pred[:, i])
                                r2_train.append(r2_train_i)
                            # 模型性能评估：测试集中的R2
                            r2_test = []
                            for i in range(y_test.shape[1]):
                                r2_test_i = metrics.r2_score(y_test.iloc[:, i], y_test_pred[:, i])
                                r2_test.append(r2_test_i)
                            # 从缓存目录中读取标准化前的数据
                            data = pd.read_excel('Cache/imputed data.xlsx',
                                                 sheet_name='filled')
                            # 转换时间格式
                            data['时间'] = pd.to_datetime(data['时间'],
                                                        format='%Y-%m-%d %H:%M:%S')
                            # 提取时间信息并将时间信息转换为索引
                            data_index = data['时间']
                            data.index = data_index
                            # 删除多余的时间列
                            data.drop(columns=['时间'], axis=1, inplace=True)
                            y_data = data[y_colums]
                            # 从缓存目录中加载标准化方法
                            with open('Cache/scaler.pkl', 'rb') as f:
                                scaler = pickle.load(f)
                            # 拟合标准化方法为反标准化做准备
                            y_scaled = scaler.fit_transform(y_data)
                            # 对输出变量的真实值和预测值进行反标准化
                            y_train = scaler.inverse_transform(y_train)
                            y_test = scaler.inverse_transform(y_test)
                            y_train_pred = scaler.inverse_transform(y_train_pred)
                            y_test_pred = scaler.inverse_transform(y_test_pred)
                            # 模型性能评估：训练集中的MAE
                            mae_train = []
                            for i in range(y_test.shape[1]):
                                mae_train_i = metrics.mean_absolute_error(y_test[:, i], y_test_pred[:, i])
                                mae_train.append(mae_train_i)
                            # 模型性能评估：测试集中的MAE
                            mae_test = []
                            for i in range(y_test.shape[1]):
                                mae_test_i = metrics.mean_absolute_error(y_test[:, i], y_test_pred[:, i])
                                mae_test.append(mae_test_i)
                            # 模型性能评估：训练集中的MSE
                            mse_train = []
                            for i in range(y_test.shape[1]):
                                mse_train_i = metrics.mean_squared_error(y_test[:, i], y_test_pred[:, i])
                                mse_train.append(mse_train_i)
                            # 模型性能评估：测试集中的MSE
                            mse_test = []
                            for i in range(y_test.shape[1]):
                                mse_test_i = metrics.mean_squared_error(y_test[:, i], y_test_pred[:, i])
                                mse_test.append(mse_test_i)
                            # 将所有结果转换为DataFrame格式以便于前端展示
                            r2_train = pd.DataFrame(r2_train, columns=['R2'])
                            r2_test = pd.DataFrame(r2_test, columns=['R2'])
                            mae_train = pd.DataFrame(mae_train, columns=['MAE'])
                            mae_test = pd.DataFrame(mae_test, columns=['MAE'])
                            mse_train = pd.DataFrame(mse_train, columns=['MSE'])
                            mse_test = pd.DataFrame(mse_test, columns=['MSE'])
                            # 汇总结果
                            train_result = pd.concat([r2_train, mae_train,
                                                      mse_train], axis=1)
                            train_result.index = y_colums
                            train_result = train_result.T
                            test_result = pd.concat([r2_test, mae_test,
                                                     mse_test], axis=1)
                            test_result.index = y_colums
                            test_result = test_result.T
                            # 点击展示训练结果
                            with st.popover("结果",
                                            use_container_width=True):
                                # 左半部分显示训练结果，右半部分显示测试结果
                                st.caption('模型训练结果如下所示：')
                                st.write(train_result)
                                
                                st.caption('模型测试结果如下所示：')
                                st.write(test_result)
                                
                                y_train_pred = pd.DataFrame(y_train_pred,
                                                            columns=y_colums)
                                st.caption('模型在训练集中的预测：')
                                st.write(y_train_pred)
                                y_test_pred = pd.DataFrame(y_test_pred,
                                                            columns=y_colums)
                                st.caption('模型在测试集中的预测：')
                                st.write(y_test_pred)
                        else:
                            with st.popover("结果",
                                            use_container_width=True):
                                st.warning('未检测到已训练的模型，请进行**训练**！', icon="⚠️")
                    # 跳转了解相关模型
                    with col21:
                        st.link_button('更多内容',
                                       "https://scikit-learn.org.cn/view/782.html",
                                       type="tertiary", icon="🔥",
                                       use_container_width=True)
                        if os.path.exists('Cache/model/svr.pkl'):
                            st.button("**已训练**", type="tertiary", icon="✅",
                                      disabled=True)
                        else:
                            st.button("**未训练**", type="tertiary", icon="⚠️",
                                      disabled=True)
def page_41():
    st.header('时间序列划分')
    with st.container(border=True):
        st.subheader('选择划分比例')
        # 检查是否指定了输入变量，若没有则提醒
        with st.spinner('检测数据中，请稍后...'):
            # 检查输入变量数据和输出变量数据是否存在
            x_data_exists = check_sheet_exists(file_path='Cache/X data.xlsx',
                                               sheet_name='X')
            y_data_exists = check_sheet_exists(file_path='Cache/y data.xlsx',
                                               sheet_name='y')
            if x_data_exists and y_data_exists:
                # 布局，将数据监测提示分成左右两部分显示
                col_31_X, col_31_y = st.columns(2)
                # 左部分提示输入变量检测成功
                with col_31_X:
                    # 从缓存目录中读取输入数据
                    X = pd.read_excel('Cache/X data.xlsx', sheet_name='X')
                    # 转换时间格式
                    X['时间'] = pd.to_datetime(X['时间'],
                                             format='%Y-%m-%d %H:%M:%S')
                    # 提取时间信息并将时间信息转换为索引
                    X_index = X['时间']
                    X.index = X_index
                    # 删除多余的时间列
                    X.drop(columns=['时间'], axis=1, inplace=True)
                    X_colums = X.columns
                    # 提示成功检测
                    st.success("已检测到输入变量数据！", icon="✅")
                # 右部分显示输出变量检测成功
                with col_31_y:
                    # 从缓存目录中读取输出数据
                    y = pd.read_excel('Cache/y data.xlsx', sheet_name='y')
                    # 转换时间格式
                    y['时间'] = pd.to_datetime(y['时间'],
                                             format='%Y-%m-%d %H:%M:%S')
                    # 提取时间信息并将时间信息转换为索引
                    y_index = y['时间']
                    y.index = y_index
                    # 删除多余的时间列
                    y.drop(columns=['时间'], axis=1, inplace=True)
                    y_colums = y.columns
                    # 提示成功检测
                    st.success("已检测到输出变量数据！", icon="✅")
                # 设置提交表单
                with st.form('feature_form'):
                    # 要求用户选择数据划分比例
                    split_ratio = st.slider("请选择数据划分比例，即训练集的占比（%）：",
                                            min_value=50, max_value=90,
                                            value=80)
                    # 设置提交表单
                    submitted = st.form_submit_button('提交')
            else:
                if not x_data_exists:
                    st.error("请选择合适的输入变量", icon="🚨")
                if not y_data_exists:
                    st.error("请选择合适的输出变量", icon="🚨")
    with st.container(border=True):
        st.subheader('训练—测试数据集')
        # 提交表单后，根据用户提供的比例划分数据集
        if submitted:
            with st.spinner('划分数据中，请稍后...'):
                # 合并输入和输出数据
                data = pd.concat([X, y], axis=1)
                data_colums = data.columns
                # 将数据划分为训练数据和测试数据
                train_size = int(split_ratio/100 * len(data))
                data_train = data[:train_size]
                data_test = data[train_size:]
                # 从训练数据和测试数据中提取输入和输出变量
                y_train = data_train[y_colums]
                X_train = data_train[X_colums]
                y_test = data_test[y_colums]
                X_test = data_test[X_colums]
                # 将所有数据储存在缓存目录中以便后续调用
                y_train.to_excel('Cache/y train data.xlsx', sheet_name='y')
                X_train.to_excel('Cache/X train data.xlsx', sheet_name='X')
                y_test.to_excel('Cache/y test data.xlsx', sheet_name='y')
                X_test.to_excel('Cache/X test data.xlsx', sheet_name='X')
                # 选择是否显示训练数据和测试数据中的输入和输出变量
                with st.expander('您的训练数据如下所示：'):
                    # 布局，将训练数据中的输入和输出变量分成左右两部分显示
                    col_X_train, col_y_train = st.columns(2)
                    # 左部分显示训练数据中的输入变量
                    with col_X_train:
                        st.caption('输入变量：')
                        st.write(X_train)
                    # 右部分显示训练数据中的输出变量
                    with col_y_train:
                       st.caption('输出变量：')
                       st.write(y_train)
                # 选择是否显示训练数据和测试数据中的输入和输出变量
                with st.expander('您的测试数据如下所示：'):
                    # 布局，将测试数据中的输入和输出变量分成左右两部分显示
                    col_X_test, col_y_test = st.columns(2)
                    # 左部分显示测试数据中的输入变量
                    with col_X_test:
                        st.caption('输入变量：')
                        st.write(X_test)
                    # 右部分显示测试数据中的输出变量
                    with col_y_test:
                        st.caption('输入变量：')
                        st.write(y_test)
        # 如果用户未提交表单，则提示
        else:
            with st.expander('您的训练数据如下所示：', expanded=True):
                st.info('请选择期望的数据划分比例，并点击**提交**！', icon="ℹ️")
            with st.expander('您的测试数据如下所示：'):
                st.info('请选择期望的数据划分比例，并点击**提交**！', icon="ℹ️")
def page_42():
    st.header('门控循环单元（GRU）')
    with st.container(border=True):
        st.subheader('超参数调优')
        with st.spinner('检测数据中，请稍后...'):
            # 检查训练数据集中的输入变量数据和输出变量数据是否存在
            x_train_exists = check_sheet_exists(file_path='Cache/X train data.xlsx',
                                                sheet_name='X')
            X_test_exists = check_sheet_exists(file_path='Cache/X test data.xlsx',
                                               sheet_name='X')
            if x_train_exists and X_test_exists:
                # 提示成功检测
                st.success("已检测到划分的训练集和测试集！请您按照流程完成训练！", icon="✅")
                # 要求用户选择超参数调优方法
                hp_option = st.selectbox("**请选择超参数调优方法：**",
                                         ("网格搜索", "随机搜索", "贝叶斯优化",
                                          "Hyperband", "人为指定"), index=4)
                if hp_option == "人为指定":
                    # 设置提交表单
                    with st.form('feature_form'):
                        # 要求用户选择时间步长
                        seq_length = st.number_input("**时间步长：**", value=7)
                        # 要求用户选择激活函数
                        act_option = st.selectbox("**隐藏层激活函数：**",
                                                  ("relu", "tanh", "sigmoid",
                                                   "linear", "exponential"))
                        # 要求用户指定随机失活
                        dropout = st.number_input("**随机失活比例：**", value=0.0,
                                                  min_value=0.0, max_value=0.8)
                        # 要求用户为提前终止法指定验证比例
                        early = st.number_input("**提前终止法验证比例：**", value=0.2,
                                                  min_value=0.1, max_value=0.5)
                        # 要求用户选择优化器
                        optimizer = st.selectbox("**优化器：**",
                                                 ("adam", "sgd", "rmsprop",
                                                  "adadelta"))
                        # 若超参数调优方法选择人工指定则要求用户选择GRU的节点数和层数
                        gru_layers = st.number_input("**隐藏层层数：**", value=1)
                        gru_units = st.number_input("**隐藏层节点数：**", value=8)
                        # 设置提交按钮
                        submitted = st.form_submit_button('确定并训练')
                # 如果使用搜索方法，则禁用一部分输入功能
                else:
                    # 设置提交表单
                    with st.form('feature_form'):
                        # 要求用户选择时间步长
                        seq_length = st.number_input("**时间步长：**", value=7)
                        # 要求用户选择激活函数
                        act_option = st.selectbox("**隐藏层激活函数：**",
                                                  ("relu", "tanh", "sigmoid",
                                                   "linear", "exponential"))
                        # 要求用户指定随机失活
                        dropout = st.number_input("**随机失活比例：**", value=0.0,
                                                  min_value=0.0, max_value=0.8)
                        # 要求用户为提前终止法指定验证比例
                        early = st.number_input("**提前中止法验证比例：**", value=0.2,
                                                  min_value=0.1, max_value=0.5)
                        # 要求用户选择优化器
                        optimizer = st.selectbox("**优化器：**",
                                                 ("adam", "sgd", "rmsprop",
                                                  "adadelta"))
                        # 若超参数调优方法选择其他方法则禁用对GRU节点数和层数的输入
                        gru_layers = st.number_input("**隐藏层层数：**",
                                                     value=1, disabled=True)
                        gru_units = st.number_input("**隐藏层节点数：**",
                                                    value=8, disabled=True)
                        # 设置提交按钮
                        submitted = st.form_submit_button('确定并训练')
            # 提示处理数据
            else:
                with st.form('feature_form'):
                    st.info('请完成数据处理！', icon="ℹ️")
                    submitted = st.form_submit_button('提交', disabled=True)
    with st.container(border=True):
        st.subheader('训练结果')
        if submitted:
            with st.spinner('训练中，请稍后...'):
                # 检查训练数据集中的输入变量数据和输出变量数据是否存在
                x_train_exists = check_sheet_exists(file_path='Cache/X train data.xlsx',
                                                    sheet_name='X')
                X_test_exists = check_sheet_exists(file_path='Cache/X test data.xlsx',
                                                   sheet_name='X')
                if x_train_exists and X_test_exists:
                    # 从缓存目录中读取输入数据
                    X_train = pd.read_excel('Cache/X train data.xlsx',
                                            sheet_name='X')
                    X_test = pd.read_excel('Cache/X test data.xlsx',
                                           sheet_name='X')
                    # 转换时间格式
                    X_train['时间'] = pd.to_datetime(X_train['时间'],
                                                   format='%Y-%m-%d %H:%M:%S')
                    X_test['时间'] = pd.to_datetime(X_test['时间'],
                                                  format='%Y-%m-%d %H:%M:%S')
                    # 提取时间信息并将时间信息转换为索引
                    X_train_index = X_train['时间']
                    X_test_index = X_test['时间']
                    X_train.index = X_train_index
                    X_test.index = X_test_index
                    # 删除多余的时间列
                    X_train.drop(columns=['时间'], axis=1, inplace=True)
                    X_test.drop(columns=['时间'], axis=1, inplace=True)
                    X_colums = X_train.columns
                    # 从缓存目录中读取输出数据
                    y_train = pd.read_excel('Cache/y train data.xlsx',
                                            sheet_name='y')
                    y_test = pd.read_excel('Cache/y test data.xlsx',
                                           sheet_name='y')
                    # 转换时间格式
                    y_train['时间'] = pd.to_datetime(y_train['时间'],
                                                   format='%Y-%m-%d %H:%M:%S')
                    y_test['时间'] = pd.to_datetime(y_test['时间'],
                                                  format='%Y-%m-%d %H:%M:%S')
                    # 提取时间信息并将时间信息转换为索引
                    y_train_index = y_train['时间']
                    y_test_index = y_test['时间']
                    y_train.index = y_train_index
                    y_test.index = y_test_index
                    # 删除多余的时间列
                    y_train.drop(columns=['时间'], axis=1, inplace=True)
                    y_test.drop(columns=['时间'], axis=1, inplace=True)
                    y_colums = y_train.columns
                    # 将DataFrame转换为Numpy数组格式以便于后续数据形式转换
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    X_test = np.array(X_test)
                    y_test = np.array(y_test)
                    # 将数据转换为序列格式
                    X_train, y_train = create_sequences(X_train, y_train,
                                                        seq_length)
                    X_test, y_test = create_sequences(X_test, y_test,
                                                      seq_length)
                    if hp_option == "人为指定":
                        gru = Sequential()
                        if gru_layers == 1:
                            gru.add(GRU(units=gru_units, activation=act_option,
                                        input_shape=(seq_length,
                                                     len(X_colums))))
                        else:
                            for i in range(gru_layers):
                                if i == 0:
                                    gru.add(GRU(units=gru_units,
                                                activation=act_option,
                                                input_shape=(seq_length,
                                                             len(X_colums)),
                                                return_sequences=True))
                                elif i == gru_layers-1:
                                    gru.add(GRU(units=gru_units,
                                                activation=act_option))
                                else:
                                    gru.add(GRU(units=gru_units,
                                                activation=act_option,
                                                dropout=dropout,
                                                return_sequences=True))
                        gru.add(Dense(units=len(y_colums)))
                        early_stop = EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                   restore_best_weights=True)
                        gru.compile(optimizer=optimizer,
                                    loss='mean_squared_error')
                        history = gru.fit(X_train, y_train, epochs=1000,
                                          validation_split=early,
                                          callbacks=[early_stop])
                        # 储存模型
                        with open('Cache/model/gru.pkl', 'wb') as f:
                            pickle.dump(gru, f)
                        # 分别在训练集和测试集中预测
                        y_train_pred = gru.predict(X_train)
                        y_test_pred = gru.predict(X_test)
                        # 将数据格式转换为DataFrame
                        hist = pd.DataFrame(history.history)
                        y_train_pred = pd.DataFrame(y_train_pred)
                        y_test_pred = pd.DataFrame(y_test_pred)
                        y_train = pd.DataFrame(y_train, columns=y_colums)
                        y_test = pd.DataFrame(y_test, columns=y_colums)
                        # 将数据储存在缓存目录以便于后续使用和计算
                        with pd.ExcelWriter('Cache/history/gru results.xlsx') as writer:
                            hist.to_excel(writer, index=False,
                                          sheet_name='hist')
                            y_train_pred.to_excel(writer, index=False,
                                                  sheet_name='trainpred')
                            y_test_pred.to_excel(writer, index=False,
                                                 sheet_name='testpred')
                            y_train.to_excel(writer, index=False,
                                             sheet_name='train')
                            y_test.to_excel(writer, index=False,
                                            sheet_name='test')
                    else:
                        def create_model(hp):
                            hp_units = hp.Int('gru_units', min_value=32,
                                              max_value=512, step=8)
                            hp_layers = hp.Int('gru_layers', min_value=1,
                                               max_value=10, step=1)
                            gru = Sequential()
                            if hp_layers == 1:
                                gru.add(GRU(units=hp_units,
                                            activation=act_option,
                                            input_shape=(seq_length,
                                                         len(X_colums))))
                            else:
                                for i in range(hp_layers):
                                    if i == 0:
                                        gru.add(GRU(units=hp_units,
                                                    activation=act_option,
                                                    input_shape=(seq_length,
                                                                 len(X_colums)),
                                                    return_sequences=True))
                                    elif i == hp_layers-1:
                                        gru.add(GRU(units=hp_units,
                                                    activation=act_option))
                                    else:
                                        gru.add(GRU(units=hp_units,
                                                    dropout=dropout,
                                                    activation=act_option,
                                                    return_sequences=True))
                            gru.add(Dense(units=len(y_colums)))
                            early_stop = EarlyStopping(monitor='val_loss',
                                                       patience=10)
                            gru.compile(optimizer=optimizer,
                                        loss='mean_squared_error')
                            return gru
                        if hp_option == "网格搜索":
                            tuner = kt.GridSearch(create_model,
                                                  objective='val_loss',
                                                  directory='Cache/hp tuning',
                                                  project_name='GRU GridSearch')
                        elif hp_option == "随机搜索":
                            tuner = kt.RandomSearch(create_model,
                                                    max_trials=1000,
                                                    objective='val_loss',
                                                    directory='Cache/hp tuning',
                                                    project_name='GRU RandomSearch')
                        elif hp_option == "贝叶斯优化":
                            tuner = kt.BayesianOptimization(create_model,
                                                            max_trials=1000,
                                                            objective='val_loss',
                                                            directory='Cache/hp tuning',
                                                            project_name='GRU Bayesian')
                        elif hp_option == "Hyperband":
                            tuner = kt.Hyperband(create_model, max_epochs=1000,
                                                 objective='val_loss',
                                                 directory='Cache/hp tuning',
                                                 project_name='GRU Hyperband')
                        early_stop = EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                   restore_best_weights=True)
                        tuner.search(X_train, y_train, validation_split=early,
                                     epochs=1000, callbacks=[early_stop])
                        best_hps = tuner.get_best_hyperparameters()[0]
                        st.write(best_hps.values)
                        best_gru = tuner.get_best_models()[0]
                        # 分别在训练集和测试集中预测
                        y_train_pred = best_gru.predict(X_train)
                        y_test_pred = best_gru.predict(X_test)
                        # 将数据格式转换为DataFrame
                        hist = pd.DataFrame(history.history)
                        y_train_pred = pd.DataFrame(y_train_pred)
                        y_test_pred = pd.DataFrame(y_test_pred)
                        y_train = pd.DataFrame(y_train, columns=y_colums)
                        y_test = pd.DataFrame(y_test, columns=y_colums)
                        # 将数据储存在缓存目录以便于后续使用和计算
                        with pd.ExcelWriter('Cache/history/gru results.xlsx') as writer:
                            hist.to_excel(writer, index=False,
                                          sheet_name='hist')
                            y_train_pred.to_excel(writer, index=False,
                                                  sheet_name='trainpred')
                            y_test_pred.to_excel(writer, index=False,
                                                 sheet_name='testpred')
                            y_train.to_excel(writer, index=False,
                                             sheet_name='train')
                            y_test.to_excel(writer, index=False,
                                            sheet_name='test')
        # 布局，将训练结果和测试结果分成左右两部分显示
        col_train, col_test = st.columns(2, gap="large")
        # 左部分显示测试数据中的输入变量
        with col_train:
            hist_exists = check_sheet_exists(file_path='Cache/history/gru results.xlsx',
                                             sheet_name='hist')
            trainpred_exists = check_sheet_exists(file_path='Cache/history/gru results.xlsx',
                                                  sheet_name='trainpred')
            testpred_exists = check_sheet_exists(file_path='Cache/history/gru results.xlsx',
                                                 sheet_name='testpred')
            train_exists = check_sheet_exists(file_path='Cache/history/gru results.xlsx',
                                              sheet_name='train')
            test_exists = check_sheet_exists(file_path='Cache/history/gru results.xlsx',
                                             sheet_name='test')
            # 检查训练历史记录是否存在
            if hist_exists:
                st.success("已检测到训练历史记录！", icon="✅")
                # 绘制训练的Loss图
                hist = pd.read_excel('Cache/history/gru results.xlsx',
                                     sheet_name='hist')
                st.caption('训练过程的Loss图：')
                st.line_chart(hist, x_label="Epoch", y_label="Loss")
                # 检查训练集的输出和预测是否存在
                if trainpred_exists and train_exists:
                    # 导入所需数据
                    y_train_pred = pd.read_excel('Cache/history/gru results.xlsx',
                                                 sheet_name='trainpred')
                    y_train = pd.read_excel('Cache/history/gru results.xlsx',
                                            sheet_name='train')
                    y_colums = y_train.columns
                    # 将DataFrame转换为Numpy数组以便于后续切片
                    y_train_pred = np.array(y_train_pred)
                    y_train = np.array(y_train)
                    # 模型性能评估：训练集中的R2
                    r2_train = []
                    for i in range(y_train.shape[1]):
                        r2_train_i = metrics.r2_score(y_train[:, i], y_train_pred[:, i])
                        r2_train.append(r2_train_i)
                    # 从缓存目录中读取标准化前的数据
                    data = pd.read_excel('Cache/imputed data.xlsx',
                                         sheet_name='filled')
                    # 转换时间格式
                    data['时间'] = pd.to_datetime(data['时间'],
                                                format='%Y-%m-%d %H:%M:%S')
                    # 提取时间信息并将时间信息转换为索引
                    data_index = data['时间']
                    data.index = data_index
                    # 删除多余的时间列并提取输出变量
                    data.drop(columns=['时间'], axis=1, inplace=True)
                    y_data = data[y_colums]
                    # 从缓存目录中加载标准化方法
                    with open('Cache/scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                    # 拟合标准化方法为反标准化做准备
                    y_scaled = scaler.fit_transform(y_data)
                    # 对输出变量的真实值和预测值进行反标准化
                    y_train = scaler.inverse_transform(y_train)
                    y_train_pred = scaler.inverse_transform(y_train_pred)
                    # 模型性能评估：训练集中的MAE
                    mae_train = []
                    for i in range(y_train.shape[1]):
                        mae_train_i = metrics.mean_absolute_error(y_train[:, i], y_train_pred[:, i])
                        mae_train.append(mae_train_i)
                    # 模型性能评估：训练集中的MSE    
                    mse_train = []
                    for i in range(y_train.shape[1]):
                        mse_train_i = metrics.mean_squared_error(y_train[:, i], y_train_pred[:, i])
                        mse_train.append(mse_train_i)
                    # 将所有结果转换为DataFrame格式以便于前端展示
                    r2_train = pd.DataFrame(r2_train, columns=['R2'])
                    mae_train = pd.DataFrame(mae_train, columns=['MAE'])
                    mse_train = pd.DataFrame(mse_train, columns=['MSE'])
                    # 汇总结果
                    train_result = pd.concat([r2_train, mae_train,
                                              mse_train], axis=1)
                    train_result.index = y_colums
                    train_result = train_result.T
                    # 在前端显示结果
                    st.caption('模型的训练结果如下所示：')
                    st.write(train_result)
                # 若训练集的输出和预测不存在，则提示
                else:
                    st.info('未检测到训练结果！请先训练模型！', icon="ℹ️")
            # 若训练历史记录不存在，则提示
            else:
                st.info('未检测到训练历史！请先训练模型！', icon="ℹ️")
        # 右部分显示测试数据中的输出变量
        with col_test:
            if testpred_exists and test_exists:
                # 绘制训练的Loss图
                st.success("已检测到模型测试结果！", icon="✅")
                st.caption('测试过程中预测值与真实值的对比：')
                y_test_pred = pd.read_excel('Cache/history/gru results.xlsx',
                                            sheet_name='testpred')
                y_test = pd.read_excel('Cache/history/gru results.xlsx',
                                       sheet_name='test')
                y_colums = y_test.columns
                y_test_pred = np.array(y_test_pred)
                y_test = np.array(y_test)
                chart_data = pd.DataFrame({"次氯酸钠预测值": y_test_pred[:, 0],
                                           "次氯酸钠真实值": y_test[:, 0],
                                           "铁盐预测值": y_test_pred[:, 1],
                                           "铁盐真实值": y_test[:, 1],
                                           "铝盐预测值": y_test_pred[:, 2],
                                           "铝盐真实值": y_test[:, 2],
                                           "硫酸铵预测值": y_test_pred[:, 3],
                                           "硫酸铵真实值": y_test[:, 3]})
                st.line_chart(chart_data, x_label="时间", y_label="标准化值")
                # 模型性能评估：测试集中的R2
                r2_test = []
                for i in range(y_test.shape[1]):
                    r2_test_i = metrics.r2_score(y_test[:, i], y_test_pred[:, i])
                    r2_test.append(r2_test_i)
                # 从缓存目录中读取标准化前的数据
                data = pd.read_excel('Cache/imputed data.xlsx',
                                     sheet_name='filled')
                # 转换时间格式
                data['时间'] = pd.to_datetime(data['时间'],
                                            format='%Y-%m-%d %H:%M:%S')
                # 提取时间信息并将时间信息转换为索引
                data_index = data['时间']
                data.index = data_index
                # 删除多余的时间列并提取输出变量
                data.drop(columns=['时间'], axis=1, inplace=True)
                y_data = data[y_colums]
                # 从缓存目录中加载标准化方法
                with open('Cache/scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                # 拟合标准化方法为反标准化做准备
                y_scaled = scaler.fit_transform(y_data)
                # 对输出变量的真实值和预测值进行反标准化
                y_test = scaler.inverse_transform(y_test)
                y_test_pred = scaler.inverse_transform(y_test_pred)
                # 模型性能评估：测试集中的MAE
                mae_test = []
                for i in range(y_test.shape[1]):
                    mae_test_i = metrics.mean_absolute_error(y_test[:, i],
                                                             y_test_pred[:, i])
                    mae_test.append(mae_test_i)
                # 模型性能评估：测试集中的MSE
                mse_test = []
                for i in range(y_test.shape[1]):
                    mse_test_i = metrics.mean_squared_error(y_test[:, i],
                                                            y_test_pred[:, i])
                    mse_test.append(mse_test_i)
                # 将所有结果转换为DataFrame格式以便于前端展示
                r2_test = pd.DataFrame(r2_test, columns=['R2'])
                mae_test = pd.DataFrame(mae_test, columns=['MAE'])
                mse_test = pd.DataFrame(mse_test, columns=['MSE'])
                # 汇总结果
                test_result = pd.concat([r2_test, mae_test,
                                         mse_test], axis=1)
                test_result.index = y_colums
                test_result = test_result.T
                # 在前端显示结果
                st.caption('模型的训练结果如下所示：')
                st.write(test_result)
            else:
                st.info('未检测到测试结果！请先训练模型！', icon="ℹ️")
            

pages = {
    "项目介绍": [
        st.Page(page_11, title="项目背景"),
        st.Page(page_12, title="注意事项"),
    ],
    "数据": [
        st.Page(page_21, title="数据上传"),
        st.Page(page_22, title="数据处理"),
        st.Page(page_23, title="特征选择")
    ],
    "机器学习方法": [
        st.Page(page_31, title="数据划分"),
        st.Page(page_32, title="模型训练"),
    ],
    "时间序列方法": [
        st.Page(page_41, title="时间序列划分"),
        st.Page(page_42, title="门控循环单元"),
    ]
}
pg = st.navigation(pages)
pg.run()