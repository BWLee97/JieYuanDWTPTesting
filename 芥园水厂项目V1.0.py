import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_authenticator as stauth
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import KNNImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
# 将页面放大至适应web宽度
st.set_page_config(layout="wide")
# 设置用户和密码
credentials = {'usernames': {
                'NKU': {'email': '593160536@qq.com',
                        'name': 'NKU', 'password': 'woshiainankaide'},
                'admin': {'email': 'likx@nankai.edu.cn',
                          'name': 'admin', 'password': 'admin'}}}
# 设置登录窗口
authenticator = stauth.Authenticate(credentials)
authenticator.login('main',
                    fields={'Form name': '基于深度学习模型的水厂智慧加药预测系统',
                            'Username': '用户名',
                            'Password': '密码',
                            'Login': '登录'})
# 判断用户登陆状态
if st.session_state['authentication_status']:
    col_a, col_b = st.columns(spec=[8, 1], vertical_alignment='bottom')
    with col_a:
        st.header('基于深度学习模型的水厂智慧加药预测系统')
    with col_b:
        authenticator.logout(button_name='退出登录')
elif st.session_state['authentication_status'] is False:
    st.error("用户名或密码不正确！", icon="🚨")
    st.stop()
elif st.session_state['authentication_status'] is None:
    st.info('请输入用户名和密码！', icon="ℹ️")
    st.stop()
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
# 定义文件夹检查函数以便于调用模型
def check_folder_empty(folder_path):
    file_list = os.listdir(folder_path)
    if len(file_list) == 0:
        return False
    else:
        return True
# 第一部分的第一页：项目背景及注意事项
def page_11():
    with st.container(border=True):
        st.subheader('项目背景')
        st.write('**简介**：本项目旨在利用深度学习模型驱动水厂投药环节的智能化精准控制，达到原水水质、水量变化的自适应最佳投药量预测，提升水厂运维管控的智能化水平和可靠程度。')
        st.write('**优势**：通过深度学习算法实现多输入多输出的高质量的预测效果，可视化网页操作方便。')
    with st.container(border=True):
        st.subheader('注意事项')
        st.write('1. 在**数据上传**页面，上传的文件格式应为.xls或xlsx；')
        st.write('2. 本系统为水厂中的时序数据提供了简便快捷的预测方法，请在待上传的Excel文件中明确指出时间列并将其命名为**时间**；')
        st.write('3. 请按侧边栏导航逐步完成**数据上传**、**数据处理**、**特征选择**和**时间序列划分**，再开始模型训练；')
        st.write('4. 在未完成模型训练前，无法在**模型应用**中进行预测。')
# 第二部分的第一页：数据上传并显示
def page_21():
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
                st.dataframe(dataframe, use_container_width=True)
        # 若未检测到上传的数据则提示上传数据
        else:
            st.caption('你上传的数据如下所示：')
            st.info('请上传训练数据！', icon="ℹ️")
# 第二部分的第二页：缺失值填充、数据标准化
def page_22():
    with st.container(border=True):
        # 设置子标题
        st.subheader('填充缺失值')
        # 检查是否上传了数据，若没有则提醒
        with st.spinner('检测数据中，请稍后...'):
            if check_sheet_exists(file_path='Cache/raw data.xlsx',
                                  sheet_name='raw'):
                # 从缓存目录中读取数据并转换格式以便于显示
                data = pd.read_excel('Cache/raw data.xlsx', sheet_name='raw')
                data['时间'] = pd.to_datetime(data['时间'],
                                            format='%Y-%m-%d %H:%M:%S')
                date_index = data['时间']
                data.index = date_index
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
                        # 设置提交表单并要求用户输入时间窗口值
                        with st.form('MAF_form'):
                            window_size = st.number_input("请输入时间窗口值：",
                                                          value=1)
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
                        # 判断缓存目录中是否存在填充数据，若有则读取并显示
                        if check_sheet_exists(file_path='Cache/imputed data.xlsx',
                                              sheet_name='filled'):
                            impute_data = pd.read_excel('Cache/imputed data.xlsx',
                                                        sheet_name='filled')
                            impute_data['时间'] = pd.to_datetime(impute_data['时间'],
                                                               format='%Y-%m-%d %H:%M:%S')
                            impute_data_index = impute_data['时间']
                            impute_data.index = impute_data_index
                            impute_data.drop(columns=['时间'], axis=1,
                                             inplace=True)
                            with st.expander('您的数据如下所示：'):
                                st.dataframe(impute_data,
                                             use_container_width=True)
                        # 提示用户完成填充
                        else:
                            with st.expander('您的数据如下所示：', expanded=True):
                                st.info('请您完成缺失值的填充！', icon="ℹ️")
                    elif missing_option == "KNN填充":
                        # 设置提交表单并要求用户指定K值
                        with st.form('KNN_form'):
                            n_neighbors = st.number_input("请输入K值：", value=1,
                                                          min_value=1,
                                                          max_value=10)
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
                        # 判断缓存目录中是否存在填充数据，若有则读取并显示
                        if check_sheet_exists(file_path='Cache/imputed data.xlsx',
                                              sheet_name='filled'):
                            impute_data = pd.read_excel('Cache/imputed data.xlsx',
                                                        sheet_name='filled')
                            impute_data['时间'] = pd.to_datetime(impute_data['时间'],
                                                               format='%Y-%m-%d %H:%M:%S')
                            impute_data_index = impute_data['时间']
                            impute_data.index = impute_data_index
                            impute_data.drop(columns=['时间'], axis=1,
                                             inplace=True)
                            with st.expander('您的数据如下所示：'):
                                st.dataframe(impute_data,
                                             use_container_width=True)
                        # 提示用户完成填充
                        else:
                            with st.expander('您的数据如下所示：', expanded=True):
                                st.info('请您完成缺失值的填充！', icon="ℹ️")
                    # 提示用户做出选择
                    else:
                        with st.expander('您的数据如下所示：', expanded=True):
                            st.info('请选择是否包含缺失值并选择填充方法！', icon="ℹ️")
                # 如果用户选择没有缺失值，则直接显示原始数据
                elif missing == "否":
                    # 将数据保存于缓存目录
                    data.to_excel('Cache/imputed data.xlsx',
                                  sheet_name='filled')
                    if check_sheet_exists(file_path='Cache/imputed data.xlsx',
                                  sheet_name='filled'):
                        # 从缓存目录中读取数据并转换格式以便于显示
                        impute_data = pd.read_excel('Cache/imputed data.xlsx',
                                                    sheet_name='filled')
                        impute_data['时间'] = pd.to_datetime(impute_data['时间'],
                                                           format='%Y-%m-%d %H:%M:%S')
                        impute_data_index = impute_data['时间']
                        impute_data.index = impute_data_index
                        impute_data.drop(columns=['时间'], axis=1, inplace=True)
                        # 选择是否显示数据
                        with st.expander('您的数据如下所示：'):
                            st.dataframe(impute_data, use_container_width=True)
                    else:
                        # 选择是否显示填充后的数据
                        with st.expander('您的数据如下所示：'):
                            st.info('请选择是否包含缺失值并选择填充方法！', icon="ℹ️")
                # 若未检测到填充后的数据则提示
                else:
                    with st.expander('您的数据如下所示：', expanded=True):
                        st.info('请选择是否包含缺失值并选择填充方法！', icon="ℹ️")
            # 若未检测到上传数据，则禁用所有选项并提示上传
            else:
                col_22_1, col_22_2 = st.columns(2)
                with col_22_1:
                    missing = st.radio("是否包含缺失值？", options=["是", "否"],
                                       index=None, disabled=True)
                with col_22_2:
                    missing_option = st.selectbox("请选择缺失值填充方法：",
                                                  ("KNN填充", "移动平均滤波"),
                                                  index=None, disabled=True,
                                                  placeholder="单击此处以选择方法")
                with st.expander('您的数据如下所示：', expanded=True):
                    st.info('请上传训练数据！', icon="ℹ️")
    with st.container(border=True):
        # 设置子标题
        st.subheader('标准化数据')
        # 检查是否填充了缺失值，若没有则提醒
        with st.spinner('检测数据中，请稍后...'):
            if check_sheet_exists(file_path='Cache/imputed data.xlsx',
                                  sheet_name='filled'):
                # 从缓存目录中读取数据并转换格式以便于操作
                data = pd.read_excel('Cache/imputed data.xlsx',
                                     sheet_name='filled')
                data['时间'] = pd.to_datetime(data['时间'],
                                            format='%Y-%m-%d %H:%M:%S')
                data_index = data['时间']
                data.index = data_index
                data.drop(columns=['时间'], axis=1, inplace=True)
                # 提示成功检测
                st.success("已检测到填充数据！", icon="✅")
                # 要求用户选择标准化方法
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
                        st.dataframe(scaled_data, use_container_width=True)
                elif norm_option == "Z值规范化":
                    # Z值规范化，其他同上
                    scaler = preprocessing.StandardScaler()
                    with open('Cache/scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    scaled_data = scaler.fit_transform(data)
                    scaled_data = pd.DataFrame(scaled_data)
                    scaled_data.index = date_index
                    scaled_data.columns = data.columns
                    scaled_data.to_excel('Cache/normalized data.xlsx',
                                         sheet_name='scaled')
                    with st.expander('您的标准化数据如下所示：'):
                        st.dataframe(scaled_data, use_container_width=True)
                elif norm_option == "鲁棒缩放":
                    # 鲁棒缩放，其他同上
                    scaler = preprocessing.RobustScaler()
                    with open('Cache/scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    scaled_data = scaler.fit_transform(data)
                    scaled_data = pd.DataFrame(scaled_data)
                    scaled_data.index = data_index
                    scaled_data.columns = data.columns
                    scaled_data.to_excel('Cache/normalized data.xlsx',
                                         sheet_name='scaled')
                    with st.expander('您的标准化数据如下所示：'):
                        st.dataframe(scaled_data, use_container_width=True)
                else:
                    with st.expander('您的标准化数据如下所示：', expanded=True):
                        st.info('请选择一种数据标准化方法！', icon="ℹ️")
            # 若未检测到填充数据，则禁用所有选项并提示上传
            else:
                norm_option = st.selectbox("请选择数据标准化方法：",
                                           ("最大最小值归一化", "Z值规范化", "鲁棒缩放"),
                                           index=None, disabled=True,
                                           placeholder="单击此处以选择方法")
                with st.expander('您的标准化数据如下所示：', expanded=True):
                    st.info('请上传并填充训练数据！', icon="ℹ️")
# 第二部分的第三页：选择输入和输出变量
def page_23():
    with st.container(border=True):
        # 设置子标题
        st.subheader('选择变量')
        # 检查是否标准化了数据值，若没有则提醒
        with st.spinner('检测数据中，请稍后...'):
            if check_sheet_exists(file_path='Cache/normalized data.xlsx',
                                  sheet_name='scaled'):
                # 从缓存目录中读取数据并转换格式以便于操作
                data = pd.read_excel('Cache/normalized data.xlsx',
                                     sheet_name='scaled')
                data['时间'] = pd.to_datetime(data['时间'],
                                            format='%Y-%m-%d %H:%M:%S')
                data_index = data['时间']
                data.index = data_index
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
                    submitted = st.form_submit_button('提交')
            # 若未检测到标准化数据，则禁用所有选项并提示处理
            else:
                st.info('请完成数据处理！', icon="ℹ️")
                with st.form('feature_form'):
                    X_options = st.multiselect("**输入变量**", ["没", "用"],
                                               placeholder='请选择模型的输出变量',
                                               disabled=True)
                    y_options = st.multiselect("**输出变量**", ["没", "用"],
                                               placeholder='请选择模型的输出变量',
                                               disabled=True)
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
            col_X, col_y = st.columns(2)
            with col_X:
                st.caption('您的输入变量如下所示：')
                st.info('请选择期望的输入变量，并点击**提交**！', icon="ℹ️")
            # 右部分显示输出变量的数据
            with col_y:
                st.caption('您的输出变量如下所示：')
                st.info('请选择期望的输出变量，并点击**提交**！', icon="ℹ️")
# 第三部分的第一页：划分训练集和测试集
def page_31():
    with st.container(border=True):
        # 设置子标题
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
                    # 从缓存目录中读取数据并转换格式以便于操作
                    X = pd.read_excel('Cache/X data.xlsx', sheet_name='X')
                    X['时间'] = pd.to_datetime(X['时间'],
                                             format='%Y-%m-%d %H:%M:%S')
                    X_index = X['时间']
                    X.index = X_index
                    X.drop(columns=['时间'], axis=1, inplace=True)
                    X_colums = X.columns
                    # 提示成功检测
                    st.success("已检测到输入变量数据！", icon="✅")
                # 右部分提示输出变量检测成功
                with col_31_y:
                    # 从缓存目录中读取数据并转换格式以便于操作
                    y = pd.read_excel('Cache/y data.xlsx', sheet_name='y')
                    y['时间'] = pd.to_datetime(y['时间'],
                                             format='%Y-%m-%d %H:%M:%S')
                    y_index = y['时间']
                    y.index = y_index
                    y.drop(columns=['时间'], axis=1, inplace=True)
                    y_colums = y.columns
                    # 提示成功检测
                    st.success("已检测到输出变量数据！", icon="✅")
                # 设置提交表单并要求用户选择数据划分比例
                with st.form('feature_form'):
                    split_ratio = st.slider("请选择数据划分比例，即训练集的占比（%）：",
                                            min_value=50, max_value=90,
                                            value=80)
                    submitted = st.form_submit_button('提交')
            # 若未检测到输入和输出数据，则禁用所有选项并提示
            else:
                col_31_X, col_31_y = st.columns(2)
                with col_31_X:
                    if not x_data_exists:
                        st.info("请选择合适的输入变量！", icon="ℹ️")
                with col_31_y:
                    if not y_data_exists:
                        st.info("请选择合适的输出变量！", icon="ℹ️")
                with st.form('feature_form'):
                    split_ratio = st.slider("请选择数据划分比例，即训练集的占比（%）：",
                                            min_value=50, max_value=90,
                                            value=80, disabled= True)
                    submitted = st.form_submit_button('提交', disabled= True)
                
    with st.container(border=True):
        # 设置子标题
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
# 第三部分的第二页：LSTM模型
def page_32():
    with st.container(border=True):
        # 设置子标题
        st.subheader('*长短期记忆神经网络（LSTM）*')
        with st.spinner('检测数据中，请稍后...'):
            # 检查数据是否存在
            X_train_exists = check_sheet_exists(file_path='Cache/X train data.xlsx',
                                                sheet_name='X')
            y_train_exists = check_sheet_exists(file_path='Cache/y train data.xlsx',
                                                sheet_name='y')
            X_test_exists = check_sheet_exists(file_path='Cache/X test data.xlsx',
                                               sheet_name='X')
            y_test_exists = check_sheet_exists(file_path='Cache/y test data.xlsx',
                                               sheet_name='y')
            if X_train_exists and X_test_exists and y_train_exists and y_test_exists:
                # 提示成功检测
                st.success("已检测到划分的训练集和测试集！请您按照流程完成训练！", icon="✅")
                # 要求用户选择超参数调优方法
                hp_option = st.selectbox("**请选择超参数调优方法：**",
                                         ("网格搜索", "随机搜索", "贝叶斯优化",
                                          "Hyperband", "人为指定"), index=4)
                # 如果选择人为指定，则激活所有功能
                if hp_option == "人为指定":
                    # 设置提交表单
                    with st.form('lstm_form'):
                        # 要求用户选择时间步长
                        seq_length = st.number_input("**时间步长：**", value=7,
                                                     min_value=1)
                        # 要求用户选择激活函数
                        act_option = st.selectbox("**隐藏层激活函数：**",
                                                  ("relu", "tanh", "sigmoid",
                                                   "linear", "exponential"))
                        # 要求用户指定随机失活
                        dropout = st.number_input("**随机失活比例：**", value=0.0,
                                                  min_value=0.0, max_value=0.8)
                        # 要求用户为提前终止法指定验证比例
                        early = st.number_input("**提前终止法验证比例：**",
                                                value=0.2, min_value=0.1,
                                                max_value=0.5)
                        # 要求用户选择优化器
                        optimizer = st.selectbox("**优化器：**",
                                                 ("adam", "sgd", "rmsprop",
                                                  "adadelta"))
                        # 要求用户指定批次大小
                        batch = st.number_input("**批次大小：**",
                                                value=32, min_value=8)
                        # 要求用户指定最大迭代次数
                        epoch = st.number_input("**最大迭代次数：**",
                                                value=1000, min_value=100)
                        # 若超参数调优方法选择人工指定则要求用户选择GRU的节点数和层数
                        lstm_layers = st.number_input("**隐藏层层数：**",
                                                      value=1, min_value=1)
                        lstm_units = st.number_input("**隐藏层节点数：**",
                                                     value=8, min_value=1)
                        # 设置提交按钮
                        submitted = st.form_submit_button('确定并训练')
                # 如果使用搜索方法，则禁用一部分功能
                else:
                    # 设置提交表单
                    with st.form('lstm_form'):
                        seq_length = st.number_input("**时间步长：**", value=7,
                                                     min_value=1)
                        act_option = st.selectbox("**隐藏层激活函数：**",
                                                  ("relu", "tanh", "sigmoid",
                                                   "linear", "exponential"))
                        dropout = st.number_input("**随机失活比例：**", value=0.0,
                                                  min_value=0.0, max_value=0.8)
                        early = st.number_input("**提前中止法验证比例：**",
                                                value=0.2, min_value=0.1,
                                                max_value=0.5)
                        optimizer = st.selectbox("**优化器：**",
                                                 ("adam", "sgd", "rmsprop",
                                                  "adadelta"))
                        batch = st.number_input("**批次大小：**",
                                                value=32, min_value=8)
                        epoch = st.number_input("**最大迭代次数：**",
                                                value=1000, min_value=100)
                        lstm_layers = st.number_input("**隐藏层层数：**",
                                                      value=1, min_value=1,
                                                      disabled=True)
                        lstm_units = st.number_input("**隐藏层节点数：**",
                                                     value=8, min_value=1,
                                                     disabled=True)
                        submitted = st.form_submit_button('确定并训练')
            # 若未检测到数据，则禁用所有选项并提示上传处理
            else:
                st.info('请完成数据处理！', icon="ℹ️")
                hp_option = st.selectbox("**请选择超参数调优方法：**",
                                         ("网格搜索", "随机搜索", "贝叶斯优化",
                                          "Hyperband", "人为指定"), index=4,
                                         disabled=True)
                with st.form('lstm_form'):
                    seq_length = st.number_input("**时间步长：**", value=7,
                                                 min_value=1, disabled=True)
                    act_option = st.selectbox("**隐藏层激活函数：**",
                                              ("relu", "tanh", "sigmoid",
                                               "linear", "exponential"),
                                              disabled=True)
                    dropout = st.number_input("**随机失活比例：**",
                                              value=0.0, min_value=0.0,
                                              max_value=0.8, disabled=True)
                    early = st.number_input("**提前中止法验证比例：**",
                                            value=0.2, min_value=0.1,
                                            max_value=0.5, disabled=True)
                    optimizer = st.selectbox("**优化器：**",
                                             ("adam", "sgd", "rmsprop",
                                              "adadelta"), disabled=True)
                    batch = st.number_input("**批次大小：**", value=32,
                                            min_value=8, disabled=True)
                    epoch = st.number_input("**最大迭代次数：**",
                                            value=1000, min_value=100,
                                            disabled=True)
                    lstm_layers = st.number_input("**隐藏层层数：**",
                                                  value=1, min_value=1,
                                                  disabled=True)
                    lstm_units = st.number_input("**隐藏层节点数：**",
                                                 value=8, min_value=1,
                                                 disabled=True)
                    submitted = st.form_submit_button('确定并训练', disabled=True)
    with st.container(border=True):
        # 设置子标题
        st.subheader('训练结果')
        if submitted:
            with st.spinner('训练中，请稍后...'):
                # 保存设置的时间步长以便于后续将新数据转换为三位张量格式
                seq_frame = pd.DataFrame({'seq_length': [seq_length]})
                seq_frame.to_excel('Cache/sequence length.xlsx', index=False)
                # 检查训练数据集中的输入变量数据和输出变量数据是否存在
                X_train_exists = check_sheet_exists(file_path='Cache/X train data.xlsx',
                                                    sheet_name='X')
                y_train_exists = check_sheet_exists(file_path='Cache/y train data.xlsx',
                                                    sheet_name='y')
                X_test_exists = check_sheet_exists(file_path='Cache/X test data.xlsx',
                                                   sheet_name='X')
                y_test_exists = check_sheet_exists(file_path='Cache/y test data.xlsx',
                                                   sheet_name='y')
                if X_train_exists and X_test_exists and y_train_exists and y_test_exists:
                    # 从缓存目录中读取输入数据并转换格式以便于操作
                    X_train = pd.read_excel('Cache/X train data.xlsx',
                                            sheet_name='X')
                    X_test = pd.read_excel('Cache/X test data.xlsx',
                                           sheet_name='X')
                    X_train['时间'] = pd.to_datetime(X_train['时间'],
                                                   format='%Y-%m-%d %H:%M:%S')
                    X_test['时间'] = pd.to_datetime(X_test['时间'],
                                                  format='%Y-%m-%d %H:%M:%S')
                    X_train_index = X_train['时间']
                    X_test_index = X_test['时间']
                    X_train.index = X_train_index
                    X_test.index = X_test_index
                    X_train.drop(columns=['时间'], axis=1, inplace=True)
                    X_test.drop(columns=['时间'], axis=1, inplace=True)
                    X_colums = X_train.columns
                    # 从缓存目录中读取输出数据并转换格式以便于操作
                    y_train = pd.read_excel('Cache/y train data.xlsx',
                                            sheet_name='y')
                    y_test = pd.read_excel('Cache/y test data.xlsx',
                                           sheet_name='y')
                    y_train['时间'] = pd.to_datetime(y_train['时间'],
                                                   format='%Y-%m-%d %H:%M:%S')
                    y_test['时间'] = pd.to_datetime(y_test['时间'],
                                                  format='%Y-%m-%d %H:%M:%S')
                    y_train_index = y_train['时间']
                    y_test_index = y_test['时间']
                    y_train.index = y_train_index
                    y_test.index = y_test_index
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
                    # 若选择人为指定超参数，则根据指定的超参数训练模型
                    if hp_option == "人为指定":
                        lstm = Sequential()
                        if lstm_layers == 1:
                            lstm.add(LSTM(units=lstm_units,
                                          activation=act_option,
                                          input_shape=(seq_length,
                                                       len(X_colums))))
                        else:
                            for i in range(lstm_layers):
                                if i == 0:
                                    lstm.add(LSTM(units=lstm_units,
                                                  activation=act_option,
                                                  input_shape=(seq_length,
                                                               len(X_colums)),
                                                  return_sequences=True))
                                elif i == lstm_layers-1:
                                    lstm.add(LSTM(units=lstm_units,
                                                  activation=act_option))
                                else:
                                    lstm.add(LSTM(units=lstm_units,
                                                  activation=act_option,
                                                  dropout=dropout,
                                                  return_sequences=True))
                        lstm.add(Dense(units=len(y_colums)))
                        early_stop = EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                   restore_best_weights=True)
                        lstm.compile(optimizer=optimizer, loss='mse')
                        history = lstm.fit(X_train, y_train, epochs=epoch,
                                           batch_size=batch,
                                           validation_split=early,
                                           callbacks=[early_stop])
                        # 储存模型
                        lstm.save('Cache/model/LSTM.keras')
                        # 分别在训练集和测试集中预测
                        y_train_pred = lstm.predict(X_train)
                        y_test_pred = lstm.predict(X_test)
                        # 将数据格式转换为DataFrame
                        hist = pd.DataFrame(history.history)
                        y_train_pred = pd.DataFrame(y_train_pred)
                        y_test_pred = pd.DataFrame(y_test_pred)
                        y_train = pd.DataFrame(y_train, columns=y_colums)
                        y_test = pd.DataFrame(y_test, columns=y_colums)
                        # 将数据储存在缓存目录以便于后续使用和计算
                        with pd.ExcelWriter('Cache/history/LSTM results.xlsx') as writer:
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
                    # 若选择超参数调优方法，则根据选择的方法搜索超参数并训练模型
                    else:
                        def create_model(hp):
                            hp_units = hp.Int('lstm_units', min_value=32,
                                              max_value=512, step=8)
                            hp_layers = hp.Int('lstm_layers', min_value=1,
                                               max_value=10, step=1)
                            lstm = Sequential()
                            if hp_layers == 1:
                                lstm.add(LSTM(units=hp_units,
                                              activation=act_option,
                                              input_shape=(seq_length,
                                                           len(X_colums))))
                            else:
                                for i in range(hp_layers):
                                    if i == 0:
                                        lstm.add(LSTM(units=hp_units,
                                                      activation=act_option,
                                                      input_shape=(seq_length,
                                                                   len(X_colums)),
                                                      return_sequences=True))
                                    elif i == hp_layers-1:
                                        lstm.add(LSTM(units=hp_units,
                                                      activation=act_option))
                                    else:
                                        lstm.add(LSTM(units=hp_units,
                                                      dropout=dropout,
                                                      activation=act_option,
                                                      return_sequences=True))
                            lstm.add(Dense(units=len(y_colums)))
                            lstm.compile(optimizer=optimizer, loss='mse')
                            return lstm
                        if hp_option == "网格搜索":
                            tuner = kt.GridSearch(create_model,
                                                  objective='val_loss',
                                                  directory='Cache/hp tuning',
                                                  project_name='LSTM GridSearch')
                        elif hp_option == "随机搜索":
                            tuner = kt.RandomSearch(create_model,
                                                    max_trials=epoch,
                                                    objective='val_loss',
                                                    directory='Cache/hp tuning',
                                                    project_name='LSTM RandomSearch')
                        elif hp_option == "贝叶斯优化":
                            tuner = kt.BayesianOptimization(create_model,
                                                            max_trials=epoch,
                                                            objective='val_loss',
                                                            directory='Cache/hp tuning',
                                                            project_name='LSTM Bayesian')
                        elif hp_option == "Hyperband":
                            tuner = kt.Hyperband(create_model, max_epochs=epoch,
                                                 objective='val_loss',
                                                 directory='Cache/hp tuning',
                                                 project_name='LSTM Hyperband')
                        early_stop = EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                   restore_best_weights=True)
                        tuner.search(X_train, y_train, validation_split=early,
                                     epochs=epoch, batch_size=batch,
                                     callbacks=[early_stop])
                        best_hps = tuner.get_best_hyperparameters()[0]
                        st.toast(best_hps.values)
                        # 获取拥有最佳超参数的模型并保存
                        best_lstm = tuner.get_best_models()[0]
                        best_lstm.save('Cache/model/LSTM.keras')
                        # 分别在训练集和测试集中预测
                        y_train_pred = best_lstm.predict(X_train)
                        y_test_pred = best_lstm.predict(X_test)
                        # 将数据格式转换为DataFrame
                        hist = pd.DataFrame(history.history)
                        y_train_pred = pd.DataFrame(y_train_pred)
                        y_test_pred = pd.DataFrame(y_test_pred)
                        y_train = pd.DataFrame(y_train, columns=y_colums)
                        y_test = pd.DataFrame(y_test, columns=y_colums)
                        # 将数据储存在缓存目录以便于后续使用和计算
                        with pd.ExcelWriter('Cache/history/LSTM results.xlsx') as writer:
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
            hist_exists = check_sheet_exists(file_path='Cache/history/LSTM results.xlsx',
                                             sheet_name='hist')
            trainpred_exists = check_sheet_exists(file_path='Cache/history/LSTM results.xlsx',
                                                  sheet_name='trainpred')
            testpred_exists = check_sheet_exists(file_path='Cache/history/LSTM results.xlsx',
                                                 sheet_name='testpred')
            train_exists = check_sheet_exists(file_path='Cache/history/LSTM results.xlsx',
                                              sheet_name='train')
            test_exists = check_sheet_exists(file_path='Cache/history/LSTM results.xlsx',
                                             sheet_name='test')
            # 检查训练历史记录是否存在
            if hist_exists:
                st.success("已检测到训练历史记录！", icon="✅")
                # 绘制训练的Loss图
                hist = pd.read_excel('Cache/history/LSTM results.xlsx',
                                     sheet_name='hist')
                st.caption('训练过程的Loss图：')
                st.line_chart(hist, x_label="Epoch", y_label="Loss")
                # 检查训练集的输出和预测是否存在
                if trainpred_exists and train_exists:
                    # 导入所需数据
                    y_train_pred = pd.read_excel('Cache/history/LSTM results.xlsx',
                                                 sheet_name='trainpred')
                    y_train = pd.read_excel('Cache/history/LSTM results.xlsx',
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
                    st.dataframe(train_result, use_container_width=True)
                # 若训练集的输出和预测不存在，则提示
                else:
                    st.caption('模型的训练结果如下所示：')
                    st.info('未检测到训练结果！', icon="ℹ️")
            # 若训练历史记录不存在，则提示
            else:
                st.caption('训练过程的Loss图：')
                st.info('未检测到训练历史！', icon="ℹ️")
                st.caption('模型的训练结果如下所示：')
                st.info('请先训练模型！', icon="ℹ️")
        # 右部分显示测试数据中的输出变量
        with col_test:
            if testpred_exists and test_exists:
                # 绘制训练的Loss图
                st.success("已检测到模型测试结果！", icon="✅")
                st.caption('测试过程中预测值与真实值的对比：')
                y_test_pred = pd.read_excel('Cache/history/LSTM results.xlsx',
                                            sheet_name='testpred')
                y_test = pd.read_excel('Cache/history/LSTM results.xlsx',
                                       sheet_name='test')
                y_colums = y_test.columns
                y_test_pred = np.array(y_test_pred)
                y_test = np.array(y_test)
                data_dict = {}
                column_names = []
                for i in range(len(y_colums)):
                    true_name = f"真实值_{i}"
                    pred_name = f"预测值_{i}"
                    column_names.extend([true_name, pred_name])
                    data_dict[true_name] = y_test[:, i]
                    data_dict[pred_name] = y_test_pred[:, i]
                    chart_data = pd.DataFrame(data_dict)
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
                st.caption('模型的测试结果如下所示：')
                st.dataframe(test_result, use_container_width=True)
            # 若测试结果记录不存在，则提示
            else:
                st.caption('测试过程中预测值与真实值的对比：')
                st.info('未检测到测试结果！', icon="ℹ️")
                st.caption('模型的训练结果如下所示：')
                st.info('请先训练模型！', icon="ℹ️")
# 第三部分的第三页：GRU模型
def page_33():
    with st.container(border=True):
        # 设置子标题
        st.subheader('*门控循环单元（GRU）*')
        with st.spinner('检测数据中，请稍后...'):
            # 检查数据是否存在
            X_train_exists = check_sheet_exists(file_path='Cache/X train data.xlsx',
                                                sheet_name='X')
            y_train_exists = check_sheet_exists(file_path='Cache/y train data.xlsx',
                                                sheet_name='y')
            X_test_exists = check_sheet_exists(file_path='Cache/X test data.xlsx',
                                               sheet_name='X')
            y_test_exists = check_sheet_exists(file_path='Cache/y test data.xlsx',
                                               sheet_name='y')
            if X_train_exists and X_test_exists and y_train_exists and y_test_exists:
                # 提示成功检测
                st.success("已检测到划分的训练集和测试集！请您按照流程完成训练！", icon="✅")
                # 要求用户选择超参数调优方法
                hp_option = st.selectbox("**请选择超参数调优方法：**",
                                         ("网格搜索", "随机搜索", "贝叶斯优化",
                                          "Hyperband", "人为指定"), index=4)
                # 如果选择人为指定，则激活所有功能
                if hp_option == "人为指定":
                    # 设置提交表单
                    with st.form('gru_form'):
                        # 要求用户选择时间步长
                        seq_length = st.number_input("**时间步长：**", value=7,
                                                     min_value=1)
                        # 要求用户选择激活函数
                        act_option = st.selectbox("**隐藏层激活函数：**",
                                                  ("relu", "tanh", "sigmoid",
                                                   "linear", "exponential"))
                        # 要求用户指定随机失活
                        dropout = st.number_input("**随机失活比例：**", value=0.0,
                                                  min_value=0.0, max_value=0.8)
                        # 要求用户为提前终止法指定验证比例
                        early = st.number_input("**提前终止法验证比例：**",
                                                value=0.2, min_value=0.1,
                                                max_value=0.5)
                        # 要求用户选择优化器
                        optimizer = st.selectbox("**优化器：**",
                                                 ("adam", "sgd", "rmsprop",
                                                  "adadelta"))
                        # 要求用户指定批次大小
                        batch = st.number_input("**批次大小：**",
                                                value=32, min_value=8)
                        # 要求用户指定最大迭代次数
                        epoch = st.number_input("**最大迭代次数：**",
                                                value=1000, min_value=100)
                        # 若超参数调优方法选择人工指定则要求用户选择GRU的节点数和层数
                        gru_layers = st.number_input("**隐藏层层数：**",
                                                     value=1, min_value=1)
                        gru_units = st.number_input("**隐藏层节点数：**",
                                                    value=8, min_value=1)
                        # 设置提交按钮
                        submitted = st.form_submit_button('确定并训练')
                # 如果使用搜索方法，则禁用一部分功能
                else:
                    # 设置提交表单
                    with st.form('gru_form'):
                        seq_length = st.number_input("**时间步长：**", value=7,
                                                     min_value=1)
                        act_option = st.selectbox("**隐藏层激活函数：**",
                                                  ("relu", "tanh", "sigmoid",
                                                   "linear", "exponential"))
                        dropout = st.number_input("**随机失活比例：**", value=0.0,
                                                  min_value=0.0, max_value=0.8)
                        early = st.number_input("**提前中止法验证比例：**",
                                                value=0.2, min_value=0.1,
                                                max_value=0.5)
                        optimizer = st.selectbox("**优化器：**",
                                                 ("adam", "sgd", "rmsprop",
                                                  "adadelta"))
                        batch = st.number_input("**批次大小：**",
                                                value=32, min_value=8)
                        epoch = st.number_input("**最大迭代次数：**",
                                                value=1000, min_value=100)
                        gru_layers = st.number_input("**隐藏层层数：**",
                                                     value=1, min_value=1,
                                                     disabled=True)
                        gru_units = st.number_input("**隐藏层节点数：**",
                                                    value=8, min_value=1,
                                                    disabled=True)
                        submitted = st.form_submit_button('确定并训练')
            # 若未检测到数据，则禁用所有选项并提示上传处理
            else:
                st.info('请完成数据处理！', icon="ℹ️")
                hp_option = st.selectbox("**请选择超参数调优方法：**",
                                         ("网格搜索", "随机搜索", "贝叶斯优化",
                                          "Hyperband", "人为指定"), index=4,
                                         disabled=True)
                with st.form('gru_form'):
                    seq_length = st.number_input("**时间步长：**", value=7,
                                                 min_value=1, disabled=True)
                    act_option = st.selectbox("**隐藏层激活函数：**",
                                              ("relu", "tanh", "sigmoid",
                                               "linear", "exponential"),
                                              disabled=True)
                    dropout = st.number_input("**随机失活比例：**",
                                              value=0.0, min_value=0.0,
                                              max_value=0.8, disabled=True)
                    early = st.number_input("**提前中止法验证比例：**",
                                            value=0.2, min_value=0.1,
                                            max_value=0.5, disabled=True)
                    optimizer = st.selectbox("**优化器：**",
                                             ("adam", "sgd", "rmsprop",
                                              "adadelta"), disabled=True)
                    batch = st.number_input("**批次大小：**", value=32,
                                            min_value=8, disabled=True)
                    epoch = st.number_input("**最大迭代次数：**",
                                            value=1000, min_value=100,
                                            disabled=True)
                    gru_layers = st.number_input("**隐藏层层数：**",
                                                 value=1, min_value=1,
                                                 disabled=True)
                    gru_units = st.number_input("**隐藏层节点数：**",
                                                value=8, min_value=1,
                                                disabled=True)
                    submitted = st.form_submit_button('确定并训练', disabled=True)
    with st.container(border=True):
        # 设置子标题
        st.subheader('训练结果')
        if submitted:
            with st.spinner('训练中，请稍后...'):
                # 保存设置的时间步长以便于后续将新数据转换为三位张量格式
                seq_frame = pd.DataFrame({'seq_length': [seq_length]})
                seq_frame.to_excel('Cache/sequence length.xlsx', index=False)
                # 检查训练数据集中的输入变量数据和输出变量数据是否存在
                X_train_exists = check_sheet_exists(file_path='Cache/X train data.xlsx',
                                                    sheet_name='X')
                y_train_exists = check_sheet_exists(file_path='Cache/y train data.xlsx',
                                                    sheet_name='y')
                X_test_exists = check_sheet_exists(file_path='Cache/X test data.xlsx',
                                                   sheet_name='X')
                y_test_exists = check_sheet_exists(file_path='Cache/y test data.xlsx',
                                                   sheet_name='y')
                if X_train_exists and X_test_exists and y_train_exists and y_test_exists:
                    # 从缓存目录中读取输入数据并转换格式以便于操作
                    X_train = pd.read_excel('Cache/X train data.xlsx',
                                            sheet_name='X')
                    X_test = pd.read_excel('Cache/X test data.xlsx',
                                           sheet_name='X')
                    X_train['时间'] = pd.to_datetime(X_train['时间'],
                                                   format='%Y-%m-%d %H:%M:%S')
                    X_test['时间'] = pd.to_datetime(X_test['时间'],
                                                  format='%Y-%m-%d %H:%M:%S')
                    X_train_index = X_train['时间']
                    X_test_index = X_test['时间']
                    X_train.index = X_train_index
                    X_test.index = X_test_index
                    X_train.drop(columns=['时间'], axis=1, inplace=True)
                    X_test.drop(columns=['时间'], axis=1, inplace=True)
                    X_colums = X_train.columns
                    # 从缓存目录中读取输出数据并转换格式以便于操作
                    y_train = pd.read_excel('Cache/y train data.xlsx',
                                            sheet_name='y')
                    y_test = pd.read_excel('Cache/y test data.xlsx',
                                           sheet_name='y')
                    y_train['时间'] = pd.to_datetime(y_train['时间'],
                                                   format='%Y-%m-%d %H:%M:%S')
                    y_test['时间'] = pd.to_datetime(y_test['时间'],
                                                  format='%Y-%m-%d %H:%M:%S')
                    y_train_index = y_train['时间']
                    y_test_index = y_test['时间']
                    y_train.index = y_train_index
                    y_test.index = y_test_index
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
                    # 若选择人为指定超参数，则根据指定的超参数训练模型
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
                        gru.compile(optimizer=optimizer, loss='mse')
                        history = gru.fit(X_train, y_train, epochs=epoch,
                                          batch_size=batch,
                                          validation_split=early,
                                          callbacks=[early_stop])
                        # 储存模型
                        gru.save('Cache/model/GRU.keras')
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
                        with pd.ExcelWriter('Cache/history/GRU results.xlsx') as writer:
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
                    # 若选择超参数调优方法，则根据选择的方法搜索超参数并训练模型
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
                            gru.compile(optimizer=optimizer, loss='mse')
                            return gru
                        if hp_option == "网格搜索":
                            tuner = kt.GridSearch(create_model,
                                                  objective='val_loss',
                                                  directory='Cache/hp tuning',
                                                  project_name='GRU GridSearch')
                        elif hp_option == "随机搜索":
                            tuner = kt.RandomSearch(create_model,
                                                    max_trials=epoch,
                                                    objective='val_loss',
                                                    directory='Cache/hp tuning',
                                                    project_name='GRU RandomSearch')
                        elif hp_option == "贝叶斯优化":
                            tuner = kt.BayesianOptimization(create_model,
                                                            max_trials=epoch,
                                                            objective='val_loss',
                                                            directory='Cache/hp tuning',
                                                            project_name='GRU Bayesian')
                        elif hp_option == "Hyperband":
                            tuner = kt.Hyperband(create_model, max_epochs=epoch,
                                                 objective='val_loss',
                                                 directory='Cache/hp tuning',
                                                 project_name='GRU Hyperband')
                        early_stop = EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                   restore_best_weights=True)
                        tuner.search(X_train, y_train, validation_split=early,
                                     epochs=epoch, batch_size=batch,
                                     callbacks=[early_stop])
                        best_hps = tuner.get_best_hyperparameters()[0]
                        st.toast(best_hps.values)
                        # 获取拥有最佳超参数的模型并保存
                        best_gru = tuner.get_best_models()[0]
                        best_gru.save('Cache/model/GRU.keras')
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
                        with pd.ExcelWriter('Cache/history/GRU results.xlsx') as writer:
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
            hist_exists = check_sheet_exists(file_path='Cache/history/GRU results.xlsx',
                                             sheet_name='hist')
            trainpred_exists = check_sheet_exists(file_path='Cache/history/GRU results.xlsx',
                                                  sheet_name='trainpred')
            testpred_exists = check_sheet_exists(file_path='Cache/history/GRU results.xlsx',
                                                 sheet_name='testpred')
            train_exists = check_sheet_exists(file_path='Cache/history/GRU results.xlsx',
                                              sheet_name='train')
            test_exists = check_sheet_exists(file_path='Cache/history/GRU results.xlsx',
                                             sheet_name='test')
            # 检查训练历史记录是否存在
            if hist_exists:
                st.success("已检测到训练历史记录！", icon="✅")
                # 绘制训练的Loss图
                hist = pd.read_excel('Cache/history/GRU results.xlsx',
                                     sheet_name='hist')
                st.caption('训练过程的Loss图：')
                st.line_chart(hist, x_label="Epoch", y_label="Loss")
                # 检查训练集的输出和预测是否存在
                if trainpred_exists and train_exists:
                    # 导入所需数据
                    y_train_pred = pd.read_excel('Cache/history/GRU results.xlsx',
                                                 sheet_name='trainpred')
                    y_train = pd.read_excel('Cache/history/GRU results.xlsx',
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
                    st.dataframe(train_result, use_container_width=True)
                # 若训练集的输出和预测不存在，则提示
                else:
                    st.caption('模型的训练结果如下所示：')
                    st.info('未检测到训练结果！', icon="ℹ️")
            # 若训练历史记录不存在，则提示
            else:
                st.caption('训练过程的Loss图：')
                st.info('未检测到训练历史！', icon="ℹ️")
                st.caption('模型的训练结果如下所示：')
                st.info('请先训练模型！', icon="ℹ️")
        # 右部分显示测试数据中的输出变量
        with col_test:
            if testpred_exists and test_exists:
                # 绘制训练的Loss图
                st.success("已检测到模型测试结果！", icon="✅")
                st.caption('测试过程中预测值与真实值的对比：')
                y_test_pred = pd.read_excel('Cache/history/GRU results.xlsx',
                                            sheet_name='testpred')
                y_test = pd.read_excel('Cache/history/GRU results.xlsx',
                                       sheet_name='test')
                y_colums = y_test.columns
                y_test_pred = np.array(y_test_pred)
                y_test = np.array(y_test)
                data_dict = {}
                column_names = []
                for i in range(len(y_colums)):
                    true_name = f"真实值_{i}"
                    pred_name = f"预测值_{i}"
                    column_names.extend([true_name, pred_name])
                    data_dict[true_name] = y_test[:, i]
                    data_dict[pred_name] = y_test_pred[:, i]
                    chart_data = pd.DataFrame(data_dict)
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
                st.dataframe(test_result, use_container_width=True)
            # 若测试结果记录不存在，则提示
            else:
                st.caption('测试过程中预测值与真实值的对比：')
                st.info('未检测到测试结果！', icon="ℹ️")
                st.caption('模型的测试结果如下所示：')
                st.info('请先训练模型！', icon="ℹ️")
# 第三部分的第四页：Seq2Seq模型
def page_34():
    with st.container(border=True):
        # 设置子标题
        st.subheader('*序列到序列模型（Seq2Seq）*')
        with st.spinner('检测数据中，请稍后...'):
            # 检查数据是否存在
            X_train_exists = check_sheet_exists(file_path='Cache/X train data.xlsx',
                                                sheet_name='X')
            y_train_exists = check_sheet_exists(file_path='Cache/y train data.xlsx',
                                                sheet_name='y')
            X_test_exists = check_sheet_exists(file_path='Cache/X test data.xlsx',
                                               sheet_name='X')
            y_test_exists = check_sheet_exists(file_path='Cache/y test data.xlsx',
                                               sheet_name='y')
            if X_train_exists and X_test_exists and y_train_exists and y_test_exists:
                # 提示成功检测
                st.success("已检测到划分的训练集和测试集！请您按照流程完成训练！", icon="✅")
                # 要求用户选择超参数调优方法
                hp_option = st.selectbox("**请选择超参数调优方法：**",
                                         ("网格搜索", "随机搜索", "贝叶斯优化",
                                          "Hyperband", "人为指定"), index=4)
                # 要求用户选择基本单元
                hp_rnn = st.selectbox("**请选择基本单元：**",
                                      ("LSTM", "GRU"), index=1)
                # 如果选择人为指定，则激活所有功能
                if hp_option == "人为指定":
                    # 设置提交表单
                    with st.form('seq_form'):
                        # 要求用户选择时间步长
                        seq_length = st.number_input("**时间步长：**", value=7,
                                                     min_value=1)
                        # 要求用户为提前终止法指定验证比例
                        early = st.number_input("**提前终止法验证比例：**",
                                                value=0.2, min_value=0.1,
                                                max_value=0.5)
                        # 要求用户选择优化器
                        optimizer = st.selectbox("**优化器：**",
                                                 ("adam", "sgd", "rmsprop",
                                                  "adadelta"))
                        # 要求用户指定批次大小
                        batch = st.number_input("**批次大小：**",
                                                value=32, min_value=8)
                        # 要求用户指定最大迭代次数
                        epoch = st.number_input("**最大迭代次数：**",
                                                value=1000, min_value=100)
                        # 若超参数调优方法选择人工指定则要求用户选择GRU的节点数和层数
                        encoder = st.number_input("**编码器层数：**", value=1,
                                                  min_value=1)
                        decoder = st.number_input("**解码器层数：**", value=1,
                                                  min_value=1)
                        units = st.number_input("**节点数：**", value=8,
                                                min_value=1)
                        # 设置提交按钮
                        submitted = st.form_submit_button('确定并训练')
                # 如果使用搜索方法，则禁用一部分功能
                else:
                    # 设置提交表单
                    with st.form('seq_form'):
                        seq_length = st.number_input("**时间步长：**", value=7,
                                                     min_value=1)
                        early = st.number_input("**提前中止法验证比例：**",
                                                value=0.2, min_value=0.1,
                                                max_value=0.5)
                        optimizer = st.selectbox("**优化器：**",
                                                 ("adam", "sgd", "rmsprop",
                                                  "adadelta"))
                        batch = st.number_input("**批次大小：**",
                                                value=32, min_value=8)
                        epoch = st.number_input("**最大迭代次数：**",
                                                value=1000, min_value=100)
                        encoder = st.number_input("**编码器层数：**", value=1,
                                                  min_value=1, disabled=True)
                        decoder = st.number_input("**解码器层数：**", value=1,
                                                  min_value=1, disabled=True)
                        units = st.number_input("**节点数：**", value=8,
                                                min_value=1, disabled=True)
                        submitted = st.form_submit_button('确定并训练')
            # 若未检测到数据，则禁用所有选项并提示上传处理
            else:
                st.info('请完成数据处理！', icon="ℹ️")
                hp_option = st.selectbox("**请选择超参数调优方法：**",
                                         ("网格搜索", "随机搜索", "贝叶斯优化",
                                          "Hyperband", "人为指定"), index=4,
                                         disabled=True)
                with st.form('seq_form'):
                    seq_length = st.number_input("**时间步长：**", value=7,
                                                 min_value=1, disabled=True)
                    early = st.number_input("**提前中止法验证比例：**",
                                            value=0.2, min_value=0.1,
                                            max_value=0.5, disabled=True)
                    optimizer = st.selectbox("**优化器：**",
                                             ("adam", "sgd", "rmsprop",
                                              "adadelta"), disabled=True)
                    batch = st.number_input("**批次大小：**", value=32,
                                            min_value=8, disabled=True)
                    epoch = st.number_input("**最大迭代次数：**",
                                            value=1000, min_value=100,
                                            disabled=True)
                    encoder = st.number_input("**编码器层数：**", value=1,
                                              min_value=1, disabled=True)
                    decoder = st.number_input("**解码器层数：**", value=1,
                                              min_value=1, disabled=True)
                    units = st.number_input("**节点数：**", value=8,
                                            min_value=1, disabled=True)
                    submitted = st.form_submit_button('确定并训练', disabled=True)
    with st.container(border=True):
        # 设置子标题
        st.subheader('训练结果')
        if submitted:
            with st.spinner('训练中，请稍后...'):
                # 保存设置的时间步长以便于后续将新数据转换为三位张量格式
                seq_frame = pd.DataFrame({'seq_length': [seq_length]})
                seq_frame.to_excel('Cache/sequence length.xlsx', index=False)
                # 检查训练数据集中的输入变量数据和输出变量数据是否存在
                X_train_exists = check_sheet_exists(file_path='Cache/X train data.xlsx',
                                                    sheet_name='X')
                y_train_exists = check_sheet_exists(file_path='Cache/y train data.xlsx',
                                                    sheet_name='y')
                X_test_exists = check_sheet_exists(file_path='Cache/X test data.xlsx',
                                                   sheet_name='X')
                y_test_exists = check_sheet_exists(file_path='Cache/y test data.xlsx',
                                                   sheet_name='y')
                if X_train_exists and X_test_exists and y_train_exists and y_test_exists:
                    # 从缓存目录中读取输入数据并转换格式以便于操作
                    X_train = pd.read_excel('Cache/X train data.xlsx',
                                            sheet_name='X')
                    X_test = pd.read_excel('Cache/X test data.xlsx',
                                           sheet_name='X')
                    X_train['时间'] = pd.to_datetime(X_train['时间'],
                                                   format='%Y-%m-%d %H:%M:%S')
                    X_test['时间'] = pd.to_datetime(X_test['时间'],
                                                  format='%Y-%m-%d %H:%M:%S')
                    X_train_index = X_train['时间']
                    X_test_index = X_test['时间']
                    X_train.index = X_train_index
                    X_test.index = X_test_index
                    X_train.drop(columns=['时间'], axis=1, inplace=True)
                    X_test.drop(columns=['时间'], axis=1, inplace=True)
                    X_colums = X_train.columns
                    # 从缓存目录中读取输出数据并转换格式以便于操作
                    y_train = pd.read_excel('Cache/y train data.xlsx',
                                            sheet_name='y')
                    y_test = pd.read_excel('Cache/y test data.xlsx',
                                           sheet_name='y')
                    y_train['时间'] = pd.to_datetime(y_train['时间'],
                                                   format='%Y-%m-%d %H:%M:%S')
                    y_test['时间'] = pd.to_datetime(y_test['时间'],
                                                  format='%Y-%m-%d %H:%M:%S')
                    y_train_index = y_train['时间']
                    y_test_index = y_test['时间']
                    y_train.index = y_train_index
                    y_test.index = y_test_index
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
                    # 若选择人为指定超参数，则根据指定的超参数训练模型
                    if hp_option == "人为指定":
                        # 若选择LSTM为基本单元
                        if hp_rnn == "LSTM":
                            encoder_input = Input(shape=(seq_length, len(X_colums)))
                            encoder_lstm = encoder_input
                            encoder_states = []
                            for i in range(encoder):
                                if i < encoder - 1:
                                    encoder_lstm, state_h, state_c = LSTM(units, return_sequences=True, return_state=True)(encoder_lstm)
                                    encoder_states.extend([state_h, state_c])
                                else:
                                    encoder_lstm, state_h, state_c = LSTM(units, return_state=True)(encoder_lstm)
                                    encoder_states.extend([state_h, state_c])
                            decoder_input = Input(shape=(1, len(y_colums)))
                            decoder_reshaped = Reshape((1, len(y_colums)))(decoder_input)
                            decoder_lstm = decoder_reshaped
                            for i in range(decoder):
                                if i < decoder - 1:
                                    decoder_lstm, state_h, state_c = LSTM(units, return_sequences=True, return_state=True)(decoder_lstm, initial_state=encoder_states[-2:] if i == 0 else None)
                                    encoder_states[-2:] = [state_h, state_c]
                                else:
                                    decoder_lstm, state_h, state_c = LSTM(units, return_sequences=True, return_state=True)(decoder_lstm, initial_state=encoder_states[-2:] if i == 0 else None)
                                    encoder_states[-2:] = [state_h, state_c]
                            decoder_output = Dense(len(y_colums))(decoder_lstm)
                        # 否则默认GRU为基本单元
                        else:
                            encoder_input = Input(shape=(seq_length, len(X_colums)))
                            encoder_gru = encoder_input
                            encoder_states = []
                            for i in range(encoder):
                                if i < encoder - 1:
                                    encoder_gru, state = GRU(units, return_sequences=True, return_state=True)(encoder_gru)
                                else:
                                    encoder_gru, state = GRU(units, return_state=True)(encoder_gru)
                                encoder_states.append(state)
                            decoder_input = Input(shape=(1, len(y_colums)))
                            decoder_reshaped = Reshape((1, len(y_colums)))(decoder_input)
                            decoder_gru = decoder_reshaped
                            for i in range(decoder):
                                if i < decoder - 1:
                                    decoder_gru, state = GRU(units, return_sequences=True, return_state=True)(decoder_gru, initial_state=encoder_states[-1] if i == 0 else None)
                                else:
                                    decoder_gru, state = GRU(units, return_sequences=True, return_state=True)(decoder_gru, initial_state=encoder_states[-1] if i == 0 else None)
                            decoder_output = Dense(len(y_colums))(decoder_gru)
                        # 建立Seq2Seq模型
                        seq = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
                        early_stop = EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                   restore_best_weights=True)
                        seq.compile(optimizer=optimizer, loss='mse')
                        # 将二维输出变量转换为三维
                        train_y = y_train.reshape(-1, 1, len(y_colums))
                        test_y = y_test.reshape(-1, 1, len(y_colums))
                        history = seq.fit([X_train, np.zeros_like(train_y)],
                                          train_y, epochs=epoch,
                                          batch_size=batch,
                                          validation_split=early,
                                          callbacks=[early_stop])
                        # 储存模型
                        seq.save('Cache/model/Seq2Seq.keras')
                        # 分别在训练集和测试集中预测
                        y_train_pred = seq.predict([X_train,
                                                    np.zeros_like(train_y)])
                        y_test_pred = seq.predict([X_test,
                                                   np.zeros_like(test_y)])
                        # 将预测结果转换为二维以便于后续计算
                        y_train_pred = y_train_pred[:, 0, :]
                        y_test_pred = y_test_pred[:, 0, :]
                        # 将数据格式转换为DataFrame
                        hist = pd.DataFrame(history.history)
                        y_train_pred = pd.DataFrame(y_train_pred)
                        y_test_pred = pd.DataFrame(y_test_pred)
                        y_train = pd.DataFrame(y_train, columns=y_colums)
                        y_test = pd.DataFrame(y_test, columns=y_colums)
                        # 将数据储存在缓存目录以便于后续使用和计算
                        with pd.ExcelWriter('Cache/history/Seq2Seq results.xlsx') as writer:
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
                    # 若选择超参数调优方法，则根据选择的方法搜索超参数并训练模型
                    else:
                        # 若选择LSTM模型则建立
                        if hp_rnn == "LSTM":
                            def create_model(hp):
                                hp_units = hp.Int('units', min_value=32,
                                                  max_value=512, step=8)
                                hp_layers_1 = hp.Int('encoder_layers',
                                                     min_value=1, max_value=10,
                                                     step=1)
                                hp_layers_2 = hp.Int('decoder_layers',
                                                     min_value=1, max_value=10,
                                                     step=1)
                                encoder_input = Input(shape=(seq_length, len(X_colums)))
                                encoder_lstm = encoder_input
                                encoder_states = []
                                for i in range(hp_layers_1):
                                    if i < encoder - 1:
                                        encoder_lstm, state_h, state_c = LSTM(hp_units, return_sequences=True, return_state=True)(encoder_lstm)
                                        encoder_states.extend([state_h, state_c])
                                    else:
                                        encoder_lstm, state_h, state_c = LSTM(hp_units, return_state=True)(encoder_lstm)
                                        encoder_states.extend([state_h, state_c])
                                decoder_input = Input(shape=(1, len(y_colums)))
                                decoder_reshaped = Reshape((1, len(y_colums)))(decoder_input)
                                decoder_lstm = decoder_reshaped
                                for i in range(hp_layers_2):
                                    if i < decoder - 1:
                                        decoder_lstm, state_h, state_c = LSTM(hp_units, return_sequences=True, return_state=True)(decoder_lstm, initial_state=encoder_states[-2:] if i == 0 else None)
                                        encoder_states[-2:] = [state_h, state_c]
                                    else:
                                        decoder_lstm, state_h, state_c = LSTM(hp_units, return_sequences=True, return_state=True)(decoder_lstm, initial_state=encoder_states[-2:] if i == 0 else None)
                                        encoder_states[-2:] = [state_h, state_c]
                                decoder_output = Dense(len(y_colums))(decoder_lstm)
                                seq = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
                                seq.compile(optimizer=optimizer, loss='mse')
                                return seq
                        # 否则建立GRU模型
                        else:
                            def create_model(hp):
                                hp_units = hp.Int('units', min_value=32,
                                                  max_value=512, step=8)
                                hp_layers_1 = hp.Int('encoder_layers',
                                                     min_value=1, max_value=10,
                                                     step=1)
                                hp_layers_2 = hp.Int('decoder_layers',
                                                     min_value=1, max_value=10,
                                                     step=1)
                                encoder_input = Input(shape=(seq_length, len(X_colums)))
                                encoder_gru = encoder_input
                                encoder_states = []
                                for i in range(hp_layers_1):
                                    if i < hp_layers_1 - 1:
                                        encoder_gru, state = GRU(hp_units, return_sequences=True, return_state=True)(encoder_gru)
                                    else:
                                        encoder_gru, state = GRU(hp_units, return_state=True)(encoder_gru)
                                    encoder_states.append(state)
                                decoder_input = Input(shape=(1, len(y_colums)))
                                decoder_reshaped = Reshape((1, len(y_colums)))(decoder_input)
                                decoder_gru = decoder_reshaped
                                for i in range(hp_layers_2):
                                    if i < hp_layers_2 - 1:
                                        decoder_gru, state = GRU(hp_units, return_sequences=True, return_state=True)(decoder_gru, initial_state=encoder_states[-1] if i == 0 else None)
                                    else:
                                        decoder_gru, state = GRU(hp_units, return_sequences=True, return_state=True)(decoder_gru, initial_state=encoder_states[-1] if i == 0 else None)
                                decoder_output = Dense(len(y_colums))(decoder_gru)
                                seq = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
                                seq.compile(optimizer=optimizer, loss='mse')
                                return seq
                            if hp_option == "网格搜索":
                                tuner = kt.GridSearch(create_model,
                                                      objective='val_loss',
                                                      directory='Cache/hp tuning',
                                                      project_name='Seq2Seq GridSearch')
                            elif hp_option == "随机搜索":
                                tuner = kt.RandomSearch(create_model,
                                                        max_trials=epoch,
                                                        objective='val_loss',
                                                        directory='Cache/hp tuning',
                                                        project_name='Seq2Seq RandomSearch')
                            elif hp_option == "贝叶斯优化":
                                tuner = kt.BayesianOptimization(create_model,
                                                                max_trials=epoch,
                                                                objective='val_loss',
                                                                directory='Cache/hp tuning',
                                                                project_name='Seq2Seq Bayesian')
                            elif hp_option == "Hyperband":
                                tuner = kt.Hyperband(create_model,
                                                     max_epochs=epoch,
                                                     objective='val_loss',
                                                     directory='Cache/hp tuning',
                                                     project_name='Seq2Seq Hyperband')
                            early_stop = EarlyStopping(monitor='val_loss',
                                                       patience=10,
                                                       restore_best_weights=True)
                            # 将二维输出变量转换为三维
                            train_y = y_train.reshape(-1, 1, len(y_colums))
                            test_y = y_test.reshape(-1, 1, len(y_colums))
                            tuner.search([X_train, np.zeros_like(train_y)],
                                         train_y, validation_split=early,
                                         epochs=epoch, batch_size=batch,
                                         callbacks=[early_stop])
                            best_hps = tuner.get_best_hyperparameters()[0]
                            st.toast(best_hps.values)
                            # 获取拥有最佳超参数的模型并保存
                            best_seq = tuner.get_best_models()[0]
                            best_seq.save('Cache/model/Seq2Seq.keras')
                            # 分别在训练集和测试集中预测
                            y_train_pred = best_seq.predict([X_train, np.zeros_like(train_y)])
                            y_test_pred = best_seq.predict([X_test, np.zeros_like(test_y)])
                            # 将预测结果转换为二维以便于后续计算
                            y_train_pred = y_train_pred[:, 0, :]
                            y_test_pred = y_test_pred[:, 0, :]
                            # 将数据格式转换为DataFrame
                            hist = pd.DataFrame(history.history)
                            y_train_pred = pd.DataFrame(y_train_pred)
                            y_test_pred = pd.DataFrame(y_test_pred)
                            y_train = pd.DataFrame(y_train, columns=y_colums)
                            y_test = pd.DataFrame(y_test, columns=y_colums)
                            # 将数据储存在缓存目录以便于后续使用和计算
                            with pd.ExcelWriter('Cache/history/Seq2Seq results.xlsx') as writer:
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
            hist_exists = check_sheet_exists(file_path='Cache/history/Seq2Seq results.xlsx',
                                             sheet_name='hist')
            trainpred_exists = check_sheet_exists(file_path='Cache/history/Seq2Seq results.xlsx',
                                                  sheet_name='trainpred')
            testpred_exists = check_sheet_exists(file_path='Cache/history/Seq2Seq results.xlsx',
                                                 sheet_name='testpred')
            train_exists = check_sheet_exists(file_path='Cache/history/Seq2Seq results.xlsx',
                                              sheet_name='train')
            test_exists = check_sheet_exists(file_path='Cache/history/Seq2Seq results.xlsx',
                                             sheet_name='test')
            # 检查训练历史记录是否存在
            if hist_exists:
                st.success("已检测到训练历史记录！", icon="✅")
                # 绘制训练的Loss图
                hist = pd.read_excel('Cache/history/Seq2Seq results.xlsx',
                                     sheet_name='hist')
                st.caption('训练过程的Loss图：')
                st.line_chart(hist, x_label="Epoch", y_label="Loss")
                # 检查训练集的输出和预测是否存在
                if trainpred_exists and train_exists:
                    # 导入所需数据
                    y_train_pred = pd.read_excel('Cache/history/Seq2Seq results.xlsx',
                                                 sheet_name='trainpred')
                    y_train = pd.read_excel('Cache/history/Seq2Seq results.xlsx',
                                            sheet_name='train')
                    y_colums = y_train.columns
                    # 将DataFrame转换为Numpy数组以便于后续切片
                    y_train_pred = np.array(y_train_pred)
                    y_train = np.array(y_train)
                    # 模型性能评估：训练集中的R2
                    r2_train = []
                    for i in range(y_train.shape[1]):
                        r2_train_i = metrics.r2_score(y_train[:, i],
                                                      y_train_pred[:, i])
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
                    st.dataframe(train_result, use_container_width=True)
                # 若训练集的输出和预测不存在，则提示
                else:
                    st.caption('模型的训练结果如下所示：')
                    st.info('未检测到训练结果！', icon="ℹ️")
            # 若训练历史记录不存在，则提示
            else:
                st.caption('训练过程的Loss图：')
                st.info('未检测到训练历史！', icon="ℹ️")
                st.caption('模型的训练结果如下所示：')
                st.info('请先训练模型！', icon="ℹ️")
        # 右部分显示测试数据中的输出变量
        with col_test:
            if testpred_exists and test_exists:
                # 绘制训练的Loss图
                st.success("已检测到模型测试结果！", icon="✅")
                st.caption('测试过程中预测值与真实值的对比：')
                y_test_pred = pd.read_excel('Cache/history/seq2seq results.xlsx',
                                            sheet_name='testpred')
                y_test = pd.read_excel('Cache/history/seq2seq results.xlsx',
                                       sheet_name='test')
                y_colums = y_test.columns
                y_test_pred = np.array(y_test_pred)
                y_test = np.array(y_test)
                data_dict = {}
                column_names = []
                for i in range(len(y_colums)):
                    true_name = f"真实值_{i}"
                    pred_name = f"预测值_{i}"
                    column_names.extend([true_name, pred_name])
                    data_dict[true_name] = y_test[:, i]
                    data_dict[pred_name] = y_test_pred[:, i]
                    chart_data = pd.DataFrame(data_dict)
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
                st.caption('模型的测试结果如下所示：')
                st.dataframe(test_result, use_container_width=True)
            # 若测试结果记录不存在，则提示
            else:
                st.caption('测试过程中预测值与真实值的对比：')
                st.info('未检测到测试结果！', icon="ℹ️")
                st.caption('模型的训练结果如下所示：')
                st.info('请先训练模型！', icon="ℹ️")
# 第四部分的第一页：做预测
def page_41():
    # 检查是否存在数据以防报错，若存在则读取并提示成功，否则提示不存在
    if os.path.exists('Cache/model'):
        model_exists = check_folder_empty('Cache/model')
    else:
        model_exists = False
    X_exists = check_sheet_exists(file_path='Cache/X data.xlsx',
                                  sheet_name='X')
    y_exists = check_sheet_exists(file_path='Cache/y data.xlsx',
                                  sheet_name='y')
    data_exists = check_sheet_exists(file_path='Cache/imputed data.xlsx',
                                     sheet_name='filled')
    if model_exists and X_exists and data_exists and y_exists:
        # 读取输入数据并提取列名
        X = pd.read_excel('Cache/X data.xlsx', sheet_name='X')
        X['时间'] = pd.to_datetime(X['时间'], format='%Y-%m-%d %H:%M:%S')
        X_index = X['时间']
        X.index = X_index
        X.drop(columns=['时间'], axis=1, inplace=True)
        X_colums = X.columns
        # 读取输出数据并提取列名
        y = pd.read_excel('Cache/y data.xlsx', sheet_name='y')
        y['时间'] = pd.to_datetime(y['时间'], format='%Y-%m-%d %H:%M:%S')
        y_index = y['时间']
        y.index = y_index
        y.drop(columns=['时间'], axis=1, inplace=True)
        y_colums = y.columns
        st.success("已检测到上传的数据和训练的模型！您可以进行**预测**操作！", icon="✅")
        # 要求用户上传待预测数据并选择期望使用的模型
        with st.form('pred_form'):
            new_file = st.file_uploader("**请上传待预测数据:**")
            st.caption("注：待预测数据的列名需要与训练模型的输入数据保持一致。")
            model_names = os.listdir('Cache/model')
            model_option = st.selectbox("**请选择模型：**", options=model_names)
            st.caption("注：模型使用三维张量格式的数据进行训练，如无特殊要求不要手动添加模型文件。")
            submitted = st.form_submit_button('预测')
    else:
        # 布局，将数据和模型的检查结果分成左右两部分显示
        col_data, col_model = st.columns(2)
        with col_data:
            if not data_exists:
                st.error('未检测到数据！', icon="🚨")
            else:
                st.success("已检测到数据！", icon="✅")
        with col_model:
            if not model_exists:
                st.error('未检测到模型！', icon="🚨")
            else:
                st.success("已检测到模型！", icon="✅")
        with st.form('pred_form'):
            new_file = st.file_uploader("**请上传待预测数据:**", disabled=True)
            st.caption("注：待预测数据的列名需要与训练模型的输入数据保持一致。")
            model_option = st.selectbox("**请选择模型：**", options='无可用模型',
                                        disabled=True)
            st.caption("注：模型使用三维张量格式的数据进行训练，请不要手动添加模型文件。")
            submitted = st.form_submit_button('预测', disabled=True)
    with st.container(border=True):
        st.subheader('预测结果')
        st.caption("模型的预测结果如下所示：")
        if submitted:
            if new_file is not None:
                # 转换为dataframe格式并提取时间信息
                X_new = pd.read_excel(new_file)
                X_new['时间'] = pd.to_datetime(X_new['时间'],
                                             format='%Y-%m-%d %H:%M:%S')
                new_index = X_new['时间']
                X_new.index = new_index
                X_new.drop(columns=['时间'], axis=1, inplace=True)
                # 从缓存目录中读取原始数据并转换格式以便于操作
                data = pd.read_excel('Cache/imputed data.xlsx',
                                     sheet_name='filled')
                data['时间'] = pd.to_datetime(data['时间'],
                                            format='%Y-%m-%d %H:%M:%S')
                data_index = data['时间']
                data.index = data_index
                data.drop(columns=['时间'], axis=1, inplace=True)
                # 提取原始数据以便于后续对新输入数据执行标准化
                X_raw = data[X_colums]
                y_raw = data[y_colums]
                # 提取用户指定的序列长度并转换为int形式
                seq_length = pd.read_excel('Cache/sequence length.xlsx')
                seq_length = seq_length['seq_length'].iloc[0]
                seq_length = int(seq_length)
                # 提取历史输入的最后几行的数据作为新数据的开头
                last_X_raw = X_raw.tail(seq_length-1)
                X_new = pd.concat([last_X_raw, X_new])
                # 检查列名是否一致，若不一致则警告
                if set(X_new.columns) != set(X_raw.columns):
                    st.error("上传的新数据不符合要求!请检查后重新上传！", icon="🚨")
                # 否则从缓存目录中加载标准化方法并对新数据执行标准化
                else:
                    with open('Cache/scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                    X_scaled = scaler.fit_transform(X_raw)
                    X_new_scaled = scaler.transform(X_new)
                    y_scaled = scaler.fit_transform(y_raw)
                    # 构造一个全零输出序列以适应构造序列函数的输入要求
                    y_zero = np.zeros((len(X_new_scaled), len(y_colums)))
                    # 将数据转换为三维张量形式
                    X_seq, y_seq = create_sequences(X_new_scaled, y_zero,
                                                    seq_length)
                    # 将二维输出变量转换为三维
                    y_seq = y_seq.reshape(-1, 1, len(y_colums))
                    # 如果选择LSTM模型，则导入模型并做出预测
                    if model_option == "LSTM.keras":
                        model = tf.keras.models.load_model('Cache/model/LSTM.keras')
                        prediction = model.predict(X_seq)
                        # 将预测结果执行反标准化并转换为DataFrame以在前端显示
                        prediction = scaler.inverse_transform(prediction)
                        prediction = pd.DataFrame(prediction, index=new_index,
                                                  columns=y_colums)
                        st.dataframe(prediction, use_container_width=True)
                    # 如果选择GRU模型，则导入模型并做出预测
                    elif model_option == "GRU.keras":
                        model = tf.keras.models.load_model('Cache/model/GRU.keras')
                        prediction = model.predict(X_seq)
                        # 将预测结果执行反标准化并转换为DataFrame以在前端显示
                        prediction = scaler.inverse_transform(prediction)
                        prediction = pd.DataFrame(prediction, index=new_index,
                                                  columns=y_colums)
                        st.dataframe(prediction, use_container_width=True)
                    # 如果选择Seq2Seq模型，则导入模型并做出预测
                    elif model_option == "Seq2Seq.keras":
                        model = tf.keras.models.load_model('Cache/model/Seq2Seq.keras')
                        prediction = model.predict([X_seq, y_seq])
                        prediction = prediction[:, 0, :]
                        # 将预测结果执行反标准化并转换为DataFrame以在前端显示
                        prediction = scaler.inverse_transform(prediction)
                        prediction = pd.DataFrame(prediction, index=new_index,
                                                  columns=y_colums)
                        st.dataframe(prediction, use_container_width=True)
                    # 以防用户混入其他模型文件
                    else:
                        st.error("请选择合适的模型！", icon="🚨")
            else:
                st.error("请上传待预测数据！", icon="🚨")
        else:
            st.info('请上传待预测数据并选择合适的模型，然后点击**预测**。', icon="ℹ️")
def page_42():
    # 检查是否存在数据以防报错，若存在则读取并展开变量输入表单
    if os.path.exists('Cache/model'):
        model_exists = check_folder_empty('Cache/model')
    else:
        model_exists = False
    X_exists = check_sheet_exists(file_path='Cache/X data.xlsx',
                                  sheet_name='X')
    y_exists = check_sheet_exists(file_path='Cache/y data.xlsx',
                                  sheet_name='y')
    data_exists = check_sheet_exists(file_path='Cache/imputed data.xlsx',
                                     sheet_name='filled')
    if model_exists and X_exists and data_exists and y_exists:
        # 读取输入数据并提取列名
        X = pd.read_excel('Cache/X data.xlsx', sheet_name='X')
        X['时间'] = pd.to_datetime(X['时间'], format='%Y-%m-%d %H:%M:%S')
        X_index = X['时间']
        X.index = X_index
        X.drop(columns=['时间'], axis=1, inplace=True)
        X_colums = X.columns
        # 读取输出数据并提取列名
        y = pd.read_excel('Cache/y data.xlsx', sheet_name='y')
        y['时间'] = pd.to_datetime(y['时间'], format='%Y-%m-%d %H:%M:%S')
        y_index = y['时间']
        y.index = y_index
        y.drop(columns=['时间'], axis=1, inplace=True)
        y_colums = y.columns
        # 从缓存目录中读取原始数据并转换格式以便于操作
        data = pd.read_excel('Cache/imputed data.xlsx',
                             sheet_name='filled')
        data['时间'] = pd.to_datetime(data['时间'],
                                    format='%Y-%m-%d %H:%M:%S')
        data_index = data['时间']
        data.index = data_index
        data.drop(columns=['时间'], axis=1, inplace=True)
        # 提取原始数据以便于后续操作
        X_raw = data[X_colums]
        y_raw = data[y_colums]
        st.success("已检测到上传的数据和训练的模型！您可以进行**预测**操作！", icon="✅")
        # 要求用户选择期望使用的模型
        model_names = os.listdir('Cache/model')
        model_option = st.selectbox("**请选择模型：**", options=model_names)
        # 设置变量输入表单
        with st.form('X_form'):
            inputs = {}
            for col in X_colums:
                min_val = X_raw[col].min()
                max_val = X_raw[col].max()
                inputs[col] = st.number_input(label=col, min_value=min_val,
                                              max_value=max_val)
            submitted = st.form_submit_button('预测')
    else:
        # 布局，将数据和模型的检查结果分成左右两部分显示
        col_data, col_model = st.columns(2)
        with col_data:
            if not data_exists:
                st.error('未检测到数据！', icon="🚨")
            else:
                st.success("已检测到数据！", icon="✅")
        with col_model:
            if not model_exists:
                st.error('未检测到模型！', icon="🚨")
            else:
                st.success("已检测到模型！", icon="✅")
        # 否则禁用所有选项并提示
        model_option = st.selectbox("**请选择模型：**", options='无可用模型',
                                    disabled=True)
        with st.form('X_form'):
            st.error('未检测到有效数据或模型！请您完成前述步骤！', icon="🚨")
            submitted = st.form_submit_button('预测', disabled=True)
    with st.container(border=True):
        st.subheader('预测结果')
        st.caption("模型的预测结果如下所示：")
        if submitted:
            inputs = pd.DataFrame(inputs, index=[0])
            # 提取用户指定的序列长度并转换为int形式
            seq_length = pd.read_excel('Cache/sequence length.xlsx')
            seq_length = seq_length['seq_length'].iloc[0]
            seq_length = int(seq_length)
            # 提取历史输入的最后几行的数据作为新数据的开头
            last_X_raw = X_raw.tail(seq_length-1)
            X_new = pd.concat([last_X_raw, inputs])
            # 否则从缓存目录中加载标准化方法并对新数据执行标准化
            with open('Cache/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            X_scaled = scaler.fit_transform(X_raw)
            X_new_scaled = scaler.transform(X_new)
            y_scaled = scaler.fit_transform(y_raw)
            # 构造一个全零输出序列以适应构造序列函数的输入要求
            y_zero = np.zeros((len(X_new_scaled), len(y_colums)))
            # 将数据转换为三维张量形式
            X_seq, y_seq = create_sequences(X_new_scaled, y_zero, seq_length)
            # 将二维输出变量转换为三维
            y_seq = y_seq.reshape(-1, 1, len(y_colums))
            # 如果选择LSTM模型，则导入模型并做出预测
            if model_option == "LSTM.keras":
                model = tf.keras.models.load_model('Cache/model/LSTM.keras')
                prediction = model.predict(X_seq)
                # 将预测结果执行反标准化并转换为DataFrame以在前端显示
                prediction = scaler.inverse_transform(prediction)
                prediction = pd.DataFrame(prediction, columns=y_colums)
                st.dataframe(prediction, hide_index=True,
                             use_container_width=True)
            # 如果选择GRU模型，则导入模型并做出预测
            elif model_option == "GRU.keras":
                model = tf.keras.models.load_model('Cache/model/GRU.keras')
                prediction = model.predict(X_seq)
                # 将预测结果执行反标准化并转换为DataFrame以在前端显示
                prediction = scaler.inverse_transform(prediction)
                prediction = pd.DataFrame(prediction, columns=y_colums)
                st.dataframe(prediction, hide_index=True,
                             use_container_width=True)
            # 如果选择Seq2Seq模型，则导入模型并做出预测
            elif model_option == "Seq2Seq.keras":
                model = tf.keras.models.load_model('Cache/model/Seq2Seq.keras')
                prediction = model.predict([X_seq, y_seq])
                prediction = prediction[:, 0, :]
                # 将预测结果执行反标准化并转换为DataFrame以在前端显示
                prediction = scaler.inverse_transform(prediction)
                prediction = pd.DataFrame(prediction, columns=y_colums)
                st.dataframe(prediction, hide_index=True,
                             use_container_width=True)
            # 以防用户混入其他模型文件
            else:
                st.error("请选择合适的模型！", icon="🚨")
        else:
            st.info('请选择合适的模型并输入相关变量值，然后点击**预测**。', icon="ℹ️")
# 侧边栏导航
pages = {"项目介绍": [st.Page(page_11, title="使用前必读")],
         "时间序列数据": [st.Page(page_21, title="数据上传"),
                    st.Page(page_22, title="数据处理"),
                    st.Page(page_23, title="特征选择")],
         "深度学习模型": [st.Page(page_31, title="时间序列划分"),
                    st.Page(page_32, title="长短期记忆神经网络"),
                    st.Page(page_33, title="门控循环单元"),
                    st.Page(page_34, title="序列到序列模型")],
         "模型应用": [st.Page(page_41, title="多步预测"),
                  st.Page(page_42, title="单步预测")]}
pg = st.navigation(pages)
pg.run()