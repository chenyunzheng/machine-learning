import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd


# 实现Bagging算法
def bagging(X, y=None, sampling_rate=0.8, iter=5, classifier_name=None):
    models = []  # trained models list
    # boostrap sampling
    for i in range(iter):
        # random train set - bootstrap sampling
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=sampling_rate)
        classifier = new_base_classifier(classifier_name)
        clf = classifier.fit(X_train, y_train)
        models.append(clf)
    print("Bagging using base classifier = '{}', End with iter = {}".format(classifier_name, iter))

    # max count as the final result (equal weighted or weighted based on vote_up/vote_all todo)
    def predict(X_test):
        y_test_predict_list = [models[i].predict_proba(X_test) for i in range(len(models))]
        # [np.argmax(np.bincount(item)) for item in np.array(y_test_predict_list).T]
        # 根据组合概率判定类别
        proba_arr = np.zeros((X_test.shape[0], 2))
        for item in y_test_predict_list:
            proba_arr += item
        return [item[1]/len(models) for item in proba_arr]

    return predict


# 实现AdaBoost.M1算法
def ada_boost_m1(X, y=None, classifiers_map=None, iter=5):
    """
    :param classifiers_map: different classifiers(map): Decision Tree, K-NN, SVM, etc
    """
    classifier_names_arr = list(classifiers_map.keys())
    cn_index = 0
    classifier_name = classifier_names_arr[cn_index]
    curr_base_classifier = classifiers_map[classifier_name]
    classifier_changed = False
    X_copy = np.copy(X)
    models = []
    coeff = []
    classifier_in_use_names = []
    weights = np.ones(X.shape[0], dtype=np.float64) / X.shape[0]  # votes_up/votes_all may be as the initial weights
    num = 0
    while num < iter:
        clf = curr_base_classifier.fit(X, y)
        y_predict = clf.predict(X)
        y_diff = (y - y_predict)  # -1/0/1
        print("==> train_predict_error = {}".format(len(weights[y_diff != 0]) / float(len(weights))))
        # calculate error ratio
        e = np.sum(weights[y_diff != 0])
        if e == 0.0:
            print(
                "Error ratio of sample classification is 0: classifier = '{}', iter = {}".format(classifier_name, num))
            e = 1e-16  # 防止零溢出
        if e >= 0.5:
            # raise Exception("iter = %d, e = %.4f, and need to change a better classifier" % (i, e))
            print(
                "ERROR >= 0.5, AdaBoost.M1 using '{}', end with e = {}, and iter = {}".format(classifier_name, e, num))
            print("Change to another type of classifier and continue...")
            cn_index = cn_index + 1
            while cn_index < len(classifier_names_arr):
                name = classifier_names_arr[cn_index]
                if name not in classifier_in_use_names:
                    break
                cn_index = cn_index + 1
            if cn_index >= len(classifier_names_arr):
                print("No available classifier")
                num = iter
                continue
            # change to another classifier and go back to fit previous X
            classifier_name = classifier_names_arr[cn_index]
            curr_base_classifier = classifiers_map[classifier_name]
            X = X_copy
            if not classifier_changed:
                num = num - 1
                classifier_changed = True
            continue
        # adjust weights
        factor = e / (1 - e)
        for index in range(len(weights)):
            weights[index] = (weights[index] * factor) if y_diff[index] == 0.0 else weights[index]
        weights /= sum(weights)
        X = np.multiply(weights.reshape(-1, 1), X)
        X_copy = np.copy(X)
        models.append(clf)
        coeff.append(np.log(1 / factor))
        # 每次拟合后，若基分类器不变，下次拟合仍需要使用基分类器，而不是由基分类器拟合数据而得到的分类器实例！！！
        # 如果使用分类器实例，则现象是得到的predict_proba一样，最终的predict概率也会一样！！！
        curr_base_classifier = new_base_classifier(classifier_name)
        classifier_in_use_names.append(classifier_name)
        classifier_changed = False
        num = num + 1
    print("AdaBoost.M1 using base classifier = '{}', End normally with iter = {}".format(
        ','.join(classifier_in_use_names), num))

    def predict(X_test):
        coeff_sum = sum(coeff)
        y_test_predict_list = [(models[i].predict_proba(X_test) * coeff[i] / coeff_sum) for i in range(len(models))]
        message = ["{}*{}".format(np.around(coeff[i] / coeff_sum, 4), classifier_in_use_names[i]) for i in
                   range(len(models))]
        proba_arr = np.zeros((X_test.shape[0], 2))
        # 根据组合概率判定类别
        for item in y_test_predict_list:
            proba_arr += item
        return [item[1] for item in proba_arr], '+'.join(message)

    return predict


def new_base_classifier(classifier_name):
    bc = None
    if classifier_name == 'SVM':
        bc = CalibratedClassifierCV(LinearSVC(dual=False))
    elif classifier_name == 'DecisionTree':
        bc = DecisionTreeClassifier()
    elif classifier_name == 'NaiveBayes':
        bc = MultinomialNB()
    elif classifier_name == 'KNN':
        bc = KNeighborsClassifier(n_neighbors=2)
    else:
        raise Exception("Unknown classifier_name = {}".format(classifier_name))
    return bc

if __name__ == '__main__':
    train_df = pd.read_csv('./data/train.csv', sep='\t')
    test_df = pd.read_csv('./data/test.csv', sep='\t')
    y_test_df = pd.read_csv('./groundTruth.csv', sep=',')
    train_df.info()
    test_df.info()
    # 根据数据格式设计特征的表示
    ### 提取特征
    max_features = 5000
    # 加载停用词
    stop_words = []
    for word in open('./stopwords-en.txt', encoding='utf8', mode='r'):
        stop_words.append(word.strip())

    # 特征值化训练集
    review_df = train_df['reviewText']
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=0.02, max_df=1.0, stop_words=stop_words)
    vectorizer.fit(review_df)
    print("text features' length =", len(vectorizer.get_feature_names_out()))
    review_feat_arr = vectorizer.transform(review_df).toarray()
    # 加入评论文本长度作为特征之一（长评论意味着高质量的概率大）
    review_length_arr = [len(item) for item in review_df]
    features_arr = np.concatenate(
        (review_feat_arr, train_df['overall'].values.reshape(-1, 1), np.reshape(review_length_arr, (-1, 1))), axis=1)
    # 归一化 ['reviewText', 'overall', 评论文本长度] 三列
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(features_arr)
    y_train = train_df['label'].values

    # 特征值化测试集
    review_df = test_df['reviewText']
    review_feat_arr = vectorizer.transform(review_df).toarray()
    # 加入评论文本长度作为特征之一（长评论意味着高质量的概率大）
    review_length_arr = [len(item) for item in review_df]
    features_arr = np.concatenate(
        (review_feat_arr, test_df['overall'].values.reshape(-1, 1), np.reshape(review_length_arr, (-1, 1))), axis=1)
    # 归一化 ['reviewText', 'overall', 评论文本长度] 三列
    X_test = scaler.transform(features_arr)
    y_test = y_test_df['Expected'].values

    ss = AdaBoostClassifier()
    ss.fit(X_train, y_train)
    print(accuracy_score(y_test, ss.predict(X_test)))
    print(roc_auc_score(y_test, ss.predict(X_test)))

    # 汇报不同组合下得到的 AUC
    ### 实例化各个分类器
    clf_dict = {
        'DecisionTree': new_base_classifier("DecisionTree"),
        'SVM': new_base_classifier("SVM"),
        'NaiveBayes': new_base_classifier("NaiveBayes"),
        # 'KNN': new_base_classifier("KNN")
    }
    for key, value in clf_dict.items():
        ## Base classifier
        base_classifier = value
        clf = base_classifier.fit(X_train, y_train)
        y_test_predict = clf.predict(X_test)
        print("Accuracy using ({}) = {}".format(key, accuracy_score(y_test, y_test_predict)))
        print("AUC using ({}) = {}".format(key, roc_auc_score(y_test, y_test_predict)))
        ## Bagging
        bagg = bagging(X_train, y_train, sampling_rate=0.8, iter=5, classifier_name=key)
        y_test_predict_bagg_proba = bagg(X_test)
        y_test_predict_bagg = [1 if item >= 0.5 else 0 for item in y_test_predict_bagg_proba]
        print("Accuracy using (Bagging + {}) = {}".format(key, accuracy_score(y_test, y_test_predict_bagg)))
        print("AUC using (Bagging + {}) = {}".format(key, roc_auc_score(y_test, y_test_predict_bagg)))
        print("---------------------------------------------------------------------")
        # 生成结果文件
        res_id_col = np.arange(1, X_test.shape[0] + 1, dtype=int).reshape(-1, 1)
        res_bagg_predict_col = np.reshape(y_test_predict_bagg_proba, (-1, 1))
        df = pd.DataFrame(np.concatenate((res_id_col, res_bagg_predict_col), axis=1), columns=['Id', 'Predicted'])
        df.to_csv("./Exp6-Bagging-{}.csv".format(key), index=False)
        ## AdaBoost.M1
        base_classifiers_map = dict()
        base_classifiers_map[key] = new_base_classifier(key)
        for k, v in clf_dict.items():
            if k != key:
                base_classifiers_map[k] = new_base_classifier(k)
        ada_boost = ada_boost_m1(X_train, y_train, iter=20, classifiers_map=base_classifiers_map)
        y_test_predict_ada_boost_proba, classifier_names = ada_boost(X_test)
        y_test_predict_ada_boost = [1 if item >= 0.5 else 0 for item in y_test_predict_ada_boost_proba]
        print("Accuracy using (AdaBoost.M1: {}) = {}".format(classifier_names,
                                                             accuracy_score(y_test, y_test_predict_ada_boost)))
        print("AUC using (AdaBoost.M1: {}) = {}".format(classifier_names,
                                                        roc_auc_score(y_test, y_test_predict_ada_boost)))
        # 生成结果文件
        res_adaboost_predict_col = np.reshape(y_test_predict_ada_boost_proba, (-1, 1))
        df = pd.DataFrame(np.concatenate((res_id_col, res_adaboost_predict_col), axis=1), columns=['Id', 'Predicted'])
        classifier_names = classifier_names.replace("*", "x")
        df.to_csv("./Exp6-AdaBoost-{}.csv".format(classifier_names), index=False)
        print("#####################################################################")
    # 结合不同集成学习算法的特点分析结果之间的差异
