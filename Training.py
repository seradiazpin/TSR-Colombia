import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import cv2
from imutils import paths
from Segmentation import Segmenter
import itertools
from sklearn.metrics import confusion_matrix


class Training:
    """
        Trainig the clasificator method
    """

    def __init__(self, sings=None, folders=None, load=False):
        if sings is not None and folders is not None:
            self.signs = sings
            self.training_paths = []
            for i in range(len(folders)):
                self.training_paths.append("./training/" + folders[i] + "/")
            self.train_lda_x = []
            self.train_lda_y = []
            self.LDA = None
            self.training_data = []
            self.SVM = None
            self.NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
            self.NNNOFIT = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
            if load:
                self.load_config()
        else:
            print ("Incluir arreglo de clases y arreglo de datos de entrenamiento\n ['clase1','clase2'],"
                   "['./folder/folderClass1', './folder/folderClass2']")

    def save_config(self):
        print("Saving config")
        joblib.dump(self.SVM, './configuration/SVM.pkl')
        joblib.dump(self.NN, './configuration/NN.plk')
        joblib.dump(self.LDA, './configuration/LDA.plk')
        joblib.dump(self.NNNOFIT, './configuration/NNNOFIT.plk')
        print("Save complete")

    def load_config(self):
        self.SVM = joblib.load('./configuration/SVM.pkl')
        self.NN = joblib.load('./configuration/NN.plk')
        self.LDA = joblib.load('./configuration/LDA.plk')
        self.NNNOFIT = joblib.load('./configuration/NNNOFIT.plk')
        print ("Load Complete")

    def train_NN(self):
        print("Training NN")
        self.NNNOFIT.fit(self.train_lda_x, self.train_lda_y)
        self.NN.fit(self.training_data, self.train_lda_y)
        print("End NN")

    def train_data(self):
        for i in range(len(self.training_paths)):
            for imagePath in paths.list_images(self.training_paths[i]):
                print (imagePath)
                s = cv2.imread(imagePath, 1)
                seg = Segmenter(s)
                seg.keypoints()
                self.train_lda_x.append(seg.descriptors())
                self.train_lda_y.append(self.signs.index(self.signs[i]))
                res = np.concatenate((seg.origi, seg.th, seg.img, seg.kpimg), axis=1)
                # cv2.imshow("img"+imagePath, s)
                # cv2.imshow("res"+imagePath, res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print (self.train_lda_x)
        print (self.train_lda_y)
        self.train_lda_x = np.array(self.train_lda_x)
        self.train_lda_y = np.array(self.train_lda_y)
        print (self.train_lda_x)
        print (self.train_lda_y)

    def draw_training(self):
        colors = ['navy', 'turquoise', 'darkorange', 'red', 'green']
        for color, i, target_name in zip(colors, [0, 1, 2, 3, 4], self.signs):
            plt.scatter(self.training_data[self.train_lda_y == i, 0], self.training_data[self.train_lda_y == i, 1],
                        alpha=.8, color=color,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('LDA of SIGN')
        plt.show()

    def train_lda(self):
        # LDA
        self.LDA = LinearDiscriminantAnalysis(n_components=2)
        self.training_data = self.LDA.fit(self.train_lda_x, self.train_lda_y).transform(self.train_lda_x)
        print (self.training_data)

    def train_SVG(self):
        print("Train SVM")
        X = self.training_data  # we only take the first two features. We could
        # avoid this ugly slicing by using a two-dim dataset
        y = self.train_lda_y

        h = .02  # step size in the mesh

        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 0.5  # SVM regularization parameter
        svc = svm.SVC(kernel='linear', C=C).fit(X, y)
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
        poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
        lin_svc = svm.LinearSVC(C=C).fit(X, y)
        self.SVM = svc
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # title for the plots
        titles = ['SVC with linear kernel',
                  'LinearSVC (linear kernel)',
                  'SVC with RBF kernel',
                  'SVC with polynomial (degree 3) kernel']

        for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            plt.subplot(2, 2, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=1)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
            plt.xlabel('Sepal length')
            plt.ylabel('Sepal width')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.title(titles[i])

        plt.show()

    def reduce_kp(self, kp):
        return self.LDA.transform(kp)

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def confusion(self, y_test, y_pred, class_names, title):
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=class_names,
                                   title=title + ', without normalization')
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                   title=title + ' Normalized confusion matrix')
        plt.show()


def test(train):
    if train:
        tra = Training(["PP", "30km", "Pare", "PEATONES"], ["Sign/pp", "Sign/V", "Sign/P", "Sign/s"])
        tra.train_data()
        tra.train_lda()
        tra.draw_training()
        tra.train_SVG()
        tra.train_NN()
        tra.save_config()
    else:
        tra = Training(["PP", "30km", "Pare", "PEATONES", "Giro"],
                       ["Sign/pp", "Sign/V", "Sign/P", "Sign/peaton", "Sign/pt"], True)
        print (tra.SVM.predict([[-1., 2.]]))


if __name__ == "__main__":
    test(True)
