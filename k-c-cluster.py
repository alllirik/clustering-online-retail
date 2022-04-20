import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_similarity

from scipy import sparse
import skfuzzy as fuzz

class KMeansClusters:
    def __init__(self, data_file_name_):
        #data file paths
        self.data_file_name = data_file_name_
        #opening files
        self.data_file =  pd.read_csv(data_file_name_) 
        
    def user_item_vect_matrix(self, users_num, popularity_inf=200, shuffle=True):
        all_users = self.data_file.visitorid.sort_values().unique()
        buying_users = self.data_file[self.data_file.event == 'transaction'].visitorid.sort_values().unique()
        viewing_users_ids = list(set(all_users) - set(buying_users))
        if shuffle:
            random.shuffle(viewing_users_ids)
        view_list = self.data_file.loc[viewing_users_ids[0:users_num]]
        view_list = view_list[view_list.event == 'view'].reset_index()
        view_list_unique = view_list.drop_duplicates('visitorid').reset_index()
        popular = self.data_file['itemid'].value_counts()
        most_popular_items = popular[popular >= popularity_inf]
        vects = np.zeros((view_list_unique.shape[0], most_popular_items.shape[0]))
        name = view_list_unique['visitorid']
        product = most_popular_items.index
        vect_matrix = pd.DataFrame(data=vects, index=name, columns=product)
        for i in range(len(view_list)):
            if view_list.loc[i]['itemid'] in vect_matrix.columns:
                vect_matrix.loc[view_list.loc[i]['visitorid'],view_list.loc[i]['itemid']] = 1
        active = vect_matrix.sum(axis=1)
        active = active[active >= 1]
        vect_matrix = vect_matrix.loc[active.index]
        return vect_matrix
    
    def scale_matrix(self, vect_matrix):
        scaler = StandardScaler()
        vect_matrix = pd.DataFrame(scaler.fit_transform(vect_matrix), 
                           index=vect_matrix.index, 
                           columns = vect_matrix.columns)
        return vect_matrix
    
    def SVD(self, vect_matrix, dim=2):
        svd = TruncatedSVD(n_components=dim)
        vect_svd = pd.DataFrame(svd.fit_transform(vect_matrix),
                                index=vect_matrix.index)
        return vect_svd
    
    def cluster_users(self, vect_svd, vect_matrix, n_clusters):
        kmeans = KMeans(n_clusters = n_clusters, init='k-means++', n_init=10)
        kmeans.fit(vect_svd)
        clusters = kmeans.predict(vect_svd)
        vect_matrix['Cluster'] = clusters
        vect_svd['Cluster'] = clusters
        (unique, counts) = np.unique(clusters, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print(frequencies)
        print(vect_svd)
        #return vect_svd['Cluster']
        return vect_matrix
    
    def neighbors(self, cluster_matrix):
        neigh = NearestNeighbors(n_neighbors=3)
        neigh.fit(cluster_matrix)
        neigh.kneighbors(cluster_matrix)
    
    def clusterization(self, n_clusters_, users_num_, popularity_inf_, scaling):
        user_item_matrix = self.user_item_vect_matrix(
            users_num = users_num_, popularity_inf = popularity_inf_)
        if scaling:
            user_item_matrix = self.scale_matrix(user_item_matrix)
        user_item_svd = self.SVD(user_item_matrix)
        clustered_users = self.cluster_users(user_item_svd, user_item_matrix, n_clusters=n_clusters_)
        print(clustered_users)
        
class KMedoidsClusters:
    def __init__(self, data_file_name_):
        #data file paths
        self.data_file_name = data_file_name_
        #opening files
        self.data_file =  pd.read_csv(data_file_name_) 
        self.userclusterlist=[]
        
    def user_item_vect_matrix(self, users_num, popularity_inf=200, shuffle=True):
        all_users = self.data_file.visitorid.sort_values().unique()
        buying_users = self.data_file[self.data_file.event == 'transaction'].visitorid.sort_values().unique()
        viewing_users_ids = list(set(all_users) - set(buying_users))
        if shuffle:
            random.shuffle(viewing_users_ids)
        view_list = self.data_file.loc[viewing_users_ids[0:users_num]]
        view_list = view_list[view_list.event == 'view'].reset_index()
        view_list_unique = view_list.drop_duplicates('visitorid').reset_index()
        popular = self.data_file['itemid'].value_counts()
        most_popular_items = popular[popular >= popularity_inf]
        vects = np.zeros((view_list_unique.shape[0], most_popular_items.shape[0]))
        name = view_list_unique['visitorid']
        product = most_popular_items.index
        vect_matrix = pd.DataFrame(data=vects, index=name, columns=product)
        for i in range(len(view_list)):
            if view_list.loc[i]['itemid'] in vect_matrix.columns:
                vect_matrix.loc[view_list.loc[i]['visitorid'],view_list.loc[i]['itemid']] = 1
        active = vect_matrix.sum(axis=1)
        active = active[active >= 1]
        vect_matrix = vect_matrix.loc[active.index]
        return vect_matrix
    
    def scale_matrix(self, vect_matrix):
        scaler = StandardScaler()
        vect_matrix = pd.DataFrame(scaler.fit_transform(vect_matrix), 
                           index=vect_matrix.index, 
                           columns = vect_matrix.columns)
        return vect_matrix
    
    def SVD(self, vect_matrix, dim=2):
        svd = TruncatedSVD(n_components=dim)
        vect_svd = pd.DataFrame(svd.fit_transform(vect_matrix),
                                index=vect_matrix.index)
        return vect_svd
    
    def cluster_users(self, vect_svd, vect_matrix, n_clusters):
        kmedoids = KMedoids(n_clusters = n_clusters, metric = 'cosine')
        kmedoids.fit(vect_svd)
        clusters = kmedoids.predict(vect_svd)
        vect_matrix['Cluster'] = clusters
        vect_svd['Cluster'] = clusters
        (unique, counts) = np.unique(clusters, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print(frequencies)
        print(vect_svd)
        #return vect_svd['Cluster']
        return vect_matrix
    
    def clusterization(self, n_clusters_, users_num_, popularity_inf_, scaling):
        user_item_matrix = self.user_item_vect_matrix(
            users_num = users_num_, popularity_inf = popularity_inf_)
        if scaling:
            user_item_matrix = self.scale_matrix(user_item_matrix)
        user_item_svd = self.SVD(user_item_matrix)
        clustered_users = self.cluster_users(user_item_svd, user_item_matrix, n_clusters=n_clusters_)
        print(clustered_users)
        self.userclusterlist = clustered_users


class CMeansClusters:
    
    def __init__(self, data_file_name_):
        #data file paths
        self.data_file_name = data_file_name_
        #opening files
        self.data_file =  pd.read_csv(data_file_name_) 
         

    def preprocess_data_v1(self, user_list, events_df):
        view_list = events_df.set_index('visitorid').loc[user_list]
        view_list = view_list[view_list.event == 'view'].reset_index()
        view_list_unique = view_list.drop_duplicates('visitorid').reset_index()

        item_list = view_list['itemid'].unique()

        vects = np.zeros((view_list_unique.shape[0], item_list.shape[0]))
        name = view_list_unique['visitorid']
        product = item_list
        vect_matrix = pd.DataFrame(data=vects, index=name, columns=product)

        for i in range(len(view_list)):
            vect_matrix.loc[view_list.loc[i]['visitorid'],view_list.loc[i]['itemid']] += 1

        return vect_matrix
    
    def user_item_matrix(self):
        
        def count_unique_views(events_df):
            dict_for_df = {}
            for event in events_df.iterrows():
                if event[1][1] not in dict_for_df:
                    dict_for_df[event[1][1]] = set()
                    dict_for_df[event[1][1]].update([event[1][3]])
                if event[1][2] == 'view':
                    dict_for_df[event[1][1]].update([event[1][3]])
            for index in dict_for_df:
                dict_for_df[index] = len(dict_for_df[index])
            return pd.DataFrame.from_dict(dict_for_df, orient='index', columns = ['num_items_viewed'])   
        
        events_df = self.data_file
        unique_view_df = count_unique_views(events_df)
        active_users = unique_view_df[unique_view_df.sum(axis=1) >= 10].index
        user_item_matrix = self.preprocess_data_v1(list(active_users), events_df)
        uim_small = user_item_matrix.loc[:, user_item_matrix.sum() >= 10]
        uim_small = uim_small[uim_small.sum(axis = 1) >= 5]
        return uim_small

    def DimRedSVD(self, vect_matrix, n):
        scaler = StandardScaler()
        vect_matrix = pd.DataFrame(scaler.fit_transform(vect_matrix), 
                               index=vect_matrix.index, 
                               columns = vect_matrix.columns)

        svd = TruncatedSVD(n_components=n)
        vect_svd = pd.DataFrame(svd.fit_transform(vect_matrix),
                                    index=vect_matrix.index)
        return vect_svd
    
    def cluster_users(self, uim, svd_matrix):
        cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
            svd_matrix.T, 25, 2, error=0.005, maxiter=1000)
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            svd_matrix.T, cntr, 2, error=0.005, maxiter=1000)
        clusters = np.argmax(u, axis=0)

        uim['Cluster'] = clusters
        (unique, counts) = np.unique(clusters, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        
        return uim
    
    def build_clusters(self):
        user_item_matrix = self.user_item_matrix()
        user_item_svd = self.DimRedSVD(user_item_matrix)
        clustered_users = self.cluster_users(user_item_matrix, user_item_svd)
        print(clustered_users)
        
    def make_binary(self, user_item_matrix):
        binary_matrix = user_item_matrix.divide(user_item_matrix, fill_value=0)
        return binary_matrix.fillna(0)

    #Similarity Matrix 
    def sim(self, data):
        
        def calculate_similarity(data):
            data_sparse = sparse.csr_matrix(data)
            similarities = cosine_similarity(data_sparse.transpose())
            sim = pd.DataFrame(data=similarities, index= data.columns, columns= data.columns)
            return sim
        
        magnitude = np.sqrt(np.square(data).sum(axis=1))
        data = data.divide(magnitude, axis='index')
        return calculate_similarity(data)
        
