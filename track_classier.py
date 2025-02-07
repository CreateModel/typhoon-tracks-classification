import pandas as pd
import numpy as np
import sklearn.metrics as sm
import glob
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.image import imread
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader
import matplotlib.lines as mlines
from sklearn.cluster import KMeans


# ___________________________________________________________________________
# -----------------------        1.数据预处理               ------------------
# ___________________________________________________________________________


# 读取ibtracs数据
df=pd.read_csv('./data/IBTrACS_usa_agency_2000-2023.csv')
# 数据来源与收集
# 数据预处理 将数据筛选台风季节（6 - 10 月）。
df['iso_time']=pd.to_datetime(df['iso_time'])
df=df[(df['iso_time'].dt.month>=6) & (df['iso_time'].dt.month<=10)]
# 筛选出台风数据 去掉name为NOT_NAMED的数据
df=df[df['name']!='NOT_NAMED']
# 删除经纬度为nan的数据
df=df.dropna(subset=['usa_lat','usa_lon'])
print('lon min-max:',df['usa_lon'].min(),df['usa_lon'].max())
print('lat min-max:',df['usa_lat'].min(),df['usa_lat'].max())
# 筛选出台风数据 选取区域为WP的数据
df=df[df['basin']=='WP']  # WP的台风有192个
df=df[['iso_time','name','sid','usa_lat','usa_lon','usa_wind', 'usa_pres','basin']]
df['year']=df['iso_time'].dt.year
# 按台风名称对数据分组
print('参与计算的台风数目：',len(df['name'].unique()))  # 515
df_group=df.groupby(['year','name'])

# 找到组里数据最多条的台风名称
max_len=0
max_len_name=None
for i in df_group:
    print(i[0])
    print(len(i[1]))
    if len(i[1])>max_len:
        max_len=len(i[1])
        max_len_name=i[0]

print(max_len_name,max_len)
# 数据处理方式1：对分组后的数据均分为20段且按照最长的台风路径进行均匀插值
# 数据处理方式2：对分组后的数据进行补零。
# 这里采用方式2进行数据处理。
# 补零
X=np.zeros((len(df_group),max_len*2)) # 台风数量，列数为台风的经度+维度长度
import PIL.Image as Image
import matplotlib.pyplot as plt

def create_map(title, extent):
    fig = plt.figure(figsize=(12, 8), dpi=400)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.imshow(
        imread('./data/NE1_50M_SR_W.tif'),
        origin='upper',
        transform=ccrs.PlateCarree(),
        extent=[-180, 180, -90, 90]
    )
    ax.set_extent(extent,crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=False, linewidth=1, color='k', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    ax.set_xticks(np.arange(extent[0], extent[1]+5, 5))
    ax.set_yticks(np.arange(extent[2], extent[3]+5, 5))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='both', labelsize=10, direction='out')

    province = shpreader.Reader('./data/Province_9.shp')
    ax.add_geometries(province.geometries(), crs=ccrs.PlateCarree(), linewidths=0.1,edgecolor='k',facecolor='none')

    # a = mlines.Line2D([],[],color='#FFFF00',marker='o',markersize=7, label='D',ls='')
    # b = mlines.Line2D([],[],color='#6495ED', marker='o',markersize=7, label='DD',ls='')
    # c = mlines.Line2D([],[],color='#3CB371', marker='o',markersize=7, label='CS',ls='')
    # d = mlines.Line2D([],[],color='#FFA500', marker='o',markersize=7, label='SCS',ls='')
    # e = mlines.Line2D([],[],color='#FF00FF', marker='o',markersize=7, label='VSCS',ls='')
    # f = mlines.Line2D([],[],color='#DC143C', marker='o',markersize=7, label='SuperCS',ls='')
    # ax.legend(handles=[a,b,c,d,e,f], numpoints=1, handletextpad=0, loc='upper left', shadow=True)
    plt.title(f'{title} Typhoon Track', fontsize=15)
    return ax

def draw_single(line,break_point):
    ax = create_map('Typhoon Track', [100, 180, 0, 55])
    for i in range(break_point):
        ax.scatter(line[i+break_point], line[i], marker='o', s=20,
                   # color=get_color(list(df['强度'])[i])
                   )

    for i in range(break_point-1):
        pointA = line[i+break_point],line[i]
        pointB = line[i+1+break_point],line[i+1]
        ax.add_geometries([sgeom.LineString([pointA, pointB])],
                          # color=get_color(list(df['强度'])[i+1]),
                          crs=ccrs.PlateCarree())
    plt.savefig('./typhoon_one.png')

def modify_index(num,index):
    new_index=index.copy()
    for i in range(num):
        min_diff=index-i
        argmin_diff=np.argmin(abs(min_diff))
        new_index[argmin_diff]=i
    return new_index

for i,j in enumerate(df_group):
    #--------------------- 补零填充方式 ----------------------
    # X[i,:len(j[1])]=j[1]['usa_lat'].values.flatten()
    # X[i,len(j[1]):len(j[1])*2]=j[1]['usa_lon'].values.flatten()
    # --------------------
    # ---------------------分段插值填充方式-----------------------
    usa_lat = pd.Series(j[1]['usa_lat'].values.flatten(), index=range(len(j[1])))
    usa_lon = pd.Series(j[1]['usa_lon'].values.flatten(), index=range(len(j[1])))

    # 创建新的索引，范围从 0 到 196，长度为 397
    new_index = np.linspace(0, len(j[1])-1, max_len)
    # 新创建的索引无整数值因此将最近索引值修正为整数值否则无法正确赋值。
    new_index = modify_index(len(usa_lat),new_index)
    # 重新索引数据，此时会产生缺失值
    interpolated_usa_lat = usa_lat.reindex(new_index)
    interpolated_usa_lon = usa_lon.reindex(new_index)

    # 使用插值方法填充缺失值，这里以线性插值为例
    interpolated_lat = interpolated_usa_lat.interpolate(method='linear')
    interpolated_lon = interpolated_usa_lon.interpolate(method='linear')
    X[i, :max_len] = interpolated_lat.values.flatten()
    X[i, max_len:max_len * 2] = interpolated_lon.values.flatten()
    # draw_single(list(X[i]), max_len)
    # --------------------------------------------------------


# 这些数据包含台风每 6 小时的位置（经度和纬度）、最低中心气压和最大持续风速。
# 为使数据适用于 FCM 模型，研究对每条台风路径进行人工插值，将其处理为等长的 20 段（），并把插值后的经度和纬度点设为列向量作为模型输入数据对象 ，即，其中（，为台风数量）
# 时间筛选：2000-2023年的数据  数据清洗

print(df.head())
print(df.tail())
print(df.describe())
print(df.columns)
print(df['name'].unique().shape)
# 提取每个台风的台风路径形成样本数据集

# ___________________________________________________________________________
# -----------------------        2.算法建模               ------------------
# ___________________________________________________________________________
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
k = 5 #聚类的类别
iteration = 500 #聚类最大循环次数
originel_x=X.copy()
# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = KMeans(n_clusters = k,n_init='auto',max_iter = iteration)
model.fit(X) #开始聚类
pred_y=model.predict(X)
score=model.score(X)

# ___________________________________________________________________________
# -----------------------        3.模型评估               ------------------
# ___________________________________________________________________________

silhouette_score=sm.silhouette_score(X, pred_y, sample_size=len(X), metric='euclidean')
print("模型准确率：",silhouette_score)

# 使用标准化后的数据进行聚类
r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
cluster_centers = pd.DataFrame(model.cluster_centers_) #找出聚类中心
cluster_centers['count'] = r1
print(cluster_centers)
result_data = pd.DataFrame(scaler.inverse_transform(X))
result_data['label'] = pred_y

# ___________________________________________________________________________
# -----------------------        4.绘制结果               ------------------
# ___________________________________________________________________________

# 绘制聚类结果
import PIL.Image as Image
import matplotlib.pyplot as plt

def create_map(title, extent):
    fig = plt.figure(figsize=(12, 8), dpi=400)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.imshow(
        imread('./data/NE1_50M_SR_W.tif'),
        origin='upper',
        transform=ccrs.PlateCarree(),
        extent=[-180, 180, -90, 90]
    )
    ax.set_extent(extent,crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=False, linewidth=1, color='k', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    ax.set_xticks(np.arange(extent[0], extent[1]+5, 5))
    ax.set_yticks(np.arange(extent[2], extent[3]+5, 5))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='both', labelsize=10, direction='out')

    province = shpreader.Reader('./data/Province_9.shp')
    ax.add_geometries(province.geometries(), crs=ccrs.PlateCarree(), linewidths=0.1,edgecolor='k',facecolor='none')

    a = mlines.Line2D([],[],color='#FFFF00',marker='o',markersize=7, label='D',ls='')
    b = mlines.Line2D([],[],color='#6495ED', marker='o',markersize=7, label='DD',ls='')
    c = mlines.Line2D([],[],color='#3CB371', marker='o',markersize=7, label='CS',ls='')
    d = mlines.Line2D([],[],color='#FFA500', marker='o',markersize=7, label='SCS',ls='')
    e = mlines.Line2D([],[],color='#FF00FF', marker='o',markersize=7, label='VSCS',ls='')
    f = mlines.Line2D([],[],color='#DC143C', marker='o',markersize=7, label='SuperCS',ls='')
    # ax.legend(handles=[a,b,c,d,e,f], numpoints=1, handletextpad=0, loc='upper left', shadow=True)
    plt.title(f'{title} Typhoon Track', fontsize=15)
    return ax



# def create_map(title, extent):
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#     url = 'http://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
#     layer = 'BlueMarble_ShadedRelief'
#     ax.add_wmts(url, layer)
#     ax.set_extent(extent,crs=ccrs.PlateCarree())
#
#     gl = ax.gridlines(draw_labels=False, linewidth=1, color='k', alpha=0.5, linestyle='--')
#     gl.xlabels_top = gl.ylabels_right = False
#     ax.set_xticks(np.arange(extent[0], extent[1]+5, 5))
#     ax.set_yticks(np.arange(extent[2], extent[3]+5, 5))
#     ax.xaxis.set_major_formatter(LongitudeFormatter())
#     ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
#     ax.yaxis.set_major_formatter(LatitudeFormatter())
#     ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
#     ax.tick_params(axis='both', labelsize=10, direction='out')
#
#     a = mlines.Line2D([],[],color='#FFFF00',marker='o',markersize=7, label='TD',ls='')
#     b = mlines.Line2D([],[],color='#6495ED', marker='o',markersize=7, label='TS',ls='')
#     c = mlines.Line2D([],[],color='#3CB371', marker='o',markersize=7, label='STS',ls='')
#     d = mlines.Line2D([],[],color='#FFA500', marker='o',markersize=7, label='TY',ls='')
#     e = mlines.Line2D([],[],color='#FF00FF', marker='o',markersize=7, label='STY',ls='')
#     f = mlines.Line2D([],[],color='#DC143C', marker='o',markersize=7, label='SSTY',ls='')
#     ax.legend(handles=[a,b,c,d,e,f], numpoints=1, handletextpad=0, loc='upper left', shadow=True)
#     plt.title(f'{title} Typhoon Track', fontsize=15)
#     return ax


for label in result_data['label'].unique():
    tc_number = result_data["label"].value_counts()[label]
    print(f'label:{label}, tc number:{tc_number}')

    one_type = result_data[result_data['label'] == label].reset_index()
    ax = create_map(f'Type {label}', [90, 160, -10, 60])
    one_type=one_type.drop(['label'],axis=1)
    for num in range(len(one_type)):
        line=one_type.iloc[num].values.tolist()
        for i in range(len(line))[:max_len]:
            ax.scatter(line[i+max_len], line[i], marker='o', s=1, color='k')

        for i in range(len(line) - 1)[:max_len]:
            pointA = line[i+max_len], line[i]
            pointB = line[i+1+max_len], line[i+1]
            ax.add_geometries([sgeom.LineString([pointA, pointB])],
                              # color=get_color(list(df['speed'])[i + 1]),
                              crs=ccrs.PlateCarree(), linewidth=1)
    plt.savefig(f'./results/track_type{label}_typhoon.png')
    plt.show()