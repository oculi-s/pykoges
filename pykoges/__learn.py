__all__ = ["modelclass"]


class softmaxResult:
    def __init__(
        self,
        scaler,
        y_test,
        prediction,
    ):
        from sklearn.metrics import confusion_matrix, accuracy_score
        import numpy as np

        self.scaler = scaler
        self.y_test = y_test
        self.prediction = prediction

        self.y_pred = np.argmax(prediction, 1)
        self.accuracy = accuracy_score(y_test, self.y_pred)
        self.conf_matrix = confusion_matrix(self.y_pred, y_test)


class modelclass:
    def __init__(
        self,
        koges,
        scalers=["minmax", "robust", "standard", "maxabs"],
    ) -> None:
        from sklearn.preprocessing import (
            MinMaxScaler,
            RobustScaler,
            StandardScaler,
            MaxAbsScaler,
        )
        import pandas as pd

        _scalers = {
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
        }
        self.koges = koges
        self.scalers = [v for k, v in _scalers.items() if k in scalers]
        self.model = None
        self.results = []

        dfs = []
        for key, df in self.koges.datas.items():
            df[self.koges.y[0]] = key
            dfs.append(df)
        if self.koges.n_class == 2:
            self.koges.data_binary = pd.concat(dfs)
        else:
            self.koges.data_multiclass = pd.concat(dfs)

    @staticmethod
    def __scale(koges, scaler):
        import pandas as pd
        from pykoges.__koges import KogesData
        from .utils import remove_duplicate_col, get_first_col

        _kg = KogesData.copy(koges)
        X = _kg.data[_kg.x].astype(float)
        Y = _kg.data[_kg.y[0]]

        X = remove_duplicate_col(X)
        Y = get_first_col(Y)

        Y = Y.reset_index(drop=True)
        scaler = scaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=_kg.x)
        _kg.data = pd.concat([X_scaled, Y], axis=1)
        return _kg

    @staticmethod
    def __split(koges):
        from .utils import remove_duplicate_col, get_first_col
        from sklearn.model_selection import train_test_split
        import random

        X = koges.data[koges.x].astype(float)
        y = koges.data[koges.y[0]]
        X = remove_duplicate_col(X)
        y = get_first_col(y)

        val_rate = 0.2
        random_state = random.randint(1, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=val_rate, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def linear(self, isdisplay=True):
        from IPython.display import display
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
        from .utils import arr_to_df, name_map

        import matplotlib.pyplot as plt
        import numpy as np

        models, r2s, results = [], [], []
        for i, scaler in enumerate(self.scalers):
            _kg = modelclass.__scale(koges=self.koges, scaler=scaler)
            X_train, X_test, y_train, y_test = modelclass.__split(_kg)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            n = len(y_test + y_train)

            # adjusted R2
            r2 = 1 - (1 - r2) * (n - 1) / (n - X_test.shape[1] - 1)

            name = model.__class__.__name__
            result = [
                ["Scaler", scaler.__name__],
                ["MAE", f"{mae:.4f}"],
                ["MSE", f"{mse:.4f}"],
                ["R2 score", f"{r2:.4f}"],
            ]
            results.append(arr_to_df(result))
            models.append(model)
            r2s.append(r2)
        r2 = max(r2s)
        model = models[r2s.index(max(r2s))]
        result = results[r2s.index(max(r2s))]
        if isdisplay:
            display(result)

        plt.ioff()
        # 입력이 하나인 경우 plot을 그립니다.
        if len(self.koges.x) == 1 and isinstance(model, LinearRegression) and isdisplay:
            plt.figure(figsize=(6, 4))
            plt.scatter(X_test, y_test, alpha=0.1)
            plt.plot(X_test, y_pred)
            plt.xlabel(self.koges.x[0])
            plt.ylabel(self.koges.y[0])
            plt.title("Regression Curve")
            plt.show()
            plt.close()

        # 요소별 중요도를 그릴 수 있는 경우 상위 8개 요소에 대한 중요도를 그립니다.
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-8:]
            features = X_test.columns
            features = [features[i] for i in indices]

            fig = plt.figure(figsize=(5, len(indices) * 0.5))
            plt.title("Feature Importances")
            plt.barh(
                range(len(indices)), importances[indices], color="b", align="center"
            )
            plt.yticks(range(len(indices)), [name_map.get(x, x) for x in features])
            plt.xlabel("Relative Importance")
            if isdisplay:
                plt.show()
            self.koges.SAVE["importance"] = fig
            plt.close()

        self.koges.SAVE[name] = result
        self.coef = model.coef_
        self.bias = model.intercept_
        self.r2 = r2

    def logistic(
        self,
        isdisplay=True,
        printCost=False,
        display_best=False,
        learning_rate=0.01,
        threshold=0.5,
        max_iter=1000000,
    ):
        from sklearn.metrics import roc_curve, auc

        from .utils import name_map
        import matplotlib.pyplot as plt
        from pykoges.__koges import KogesData
        import numpy as np

        if not hasattr(self.koges, "data_binary"):
            raise ValueError("binary dataset이 정의되지 않았습니다.")

        y = self.koges.y[0]
        plt.ioff()
        _kg = KogesData.copy(self.koges)

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def compute_cost(X, y, W):
            m = len(y)
            h = sigmoid(X.dot(W))
            cost = -1 / m * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))
            return cost

        models, roc_aucs, fprs, tprs = [], [], [], []
        for i, scaler in enumerate(self.scalers):
            _kg.data = _kg.data_binary
            _kg.data[y] = _kg.data[y].astype(int)

            _kg = modelclass.__scale(_kg, scaler=scaler)
            X_train, X_test, y_train, y_test = modelclass.__split(koges=_kg)

            # add bias
            X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
            X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

            cnt = 0
            _W = np.random.randn(X_train.shape[1])
            if printCost:
                print(scaler.__name__)
            while cnt < max_iter:
                h = sigmoid(X_train.dot(_W))
                grad = X_train.T.dot(h - y_train) / len(y)
                _W -= learning_rate * grad
                cost = compute_cost(X_train, y_train, _W)
                if printCost and cnt % 10000 == 0:
                    print(f"iter: {cnt} cost: {cost}")
                if cost < threshold:
                    break
                cnt += 1

            y_pred_prob = sigmoid(X_test.dot(_W))
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            fprs.append(fpr)
            tprs.append(tpr)
            roc_auc = auc(fpr, tpr)

            models.append(_W)
            roc_aucs.append(roc_auc)

        y_name = name_map.get(y, y)
        if display_best:
            best_idx = roc_aucs.index(max(roc_aucs))

            fig = plt.figure(figsize=(5, 4))
            plt.plot(
                fprs[best_idx],
                tprs[best_idx],
                color="grey",
                lw=2,
                label=f"ROC curve (auc= {roc_aucs[best_idx]:.2f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.title(f"{y_name} ({self.scalers[best_idx].__name__})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            self.koges.SAVE["LogisticRegression"] = fig
        else:
            ncol = len(self.scalers)
            fig, ax = plt.subplots(
                nrows=1,
                ncols=ncol,
                figsize=(ncol * 3, 3.5),
                constrained_layout=True,
                sharey=True,
            )
            for i, scaler in enumerate(self.scalers):
                plt.subplot(1, ncol, i + 1)
                plt.plot(
                    fprs[i],
                    tprs[i],
                    color="grey",
                    lw=2,
                    label=f"ROC curve (auc = {roc_aucs[i]:.2f})",
                )
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.title(f"{y_name} ({scaler.__name__})")
                plt.legend(loc="lower right")

                fig.supxlabel("False Positive Rate")
                fig.supylabel("True Positive Rate")
                fig.suptitle("ROC curve")
            self.koges.SAVE["LogisticRegression"] = fig
        if isdisplay:
            plt.show()
        plt.close()

        roc_auc = max(roc_aucs)
        _W = models[roc_aucs.index(roc_auc)]
        self.bias, self.coef = _W[0], _W[1:]
        self.roc_auc = roc_auc

    def __muticlassRoc(self, result, isdisplay=True):
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt

        aucs = []
        for i, _class in enumerate(self.koges.classes):
            if (result.y_test == i).sum():
                fpr, tpr, _ = roc_curve(result.y_test == i, result.prediction[:, i])
                roc_auc = auc(fpr, tpr)
                if isdisplay:
                    plt.plot(
                        fpr,
                        tpr,
                        lw=2,
                        label=f"{_class} (auc = {roc_auc:.2f})",
                        color="b",
                        alpha=(1 - i * 0.2),
                    )
            else:
                roc_auc = 0
            aucs.append(roc_auc)
        if isdisplay:
            plt.plot([0, 1], [0, 1], "k--", lw=2)
            plt.legend(loc="lower right")
        return aucs

    def __get_best(self):
        auc = 0
        best = False
        for r in self.results:
            aucs = self.__muticlassRoc(result=r, isdisplay=False)
            auc_mean = sum(aucs) / len(aucs)
            if auc_mean > auc:
                auc = auc_mean
                best = r
        return best

    def softmax(
        self,
        display_roc_curve=True,
        display_confusion_matrix=True,
        display_best=False,
        printCost=False,
        learning_rate=0.01,
        threshold=0.5,
        max_iter=1000000,
    ):
        from pykoges.__koges import KogesData

        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        sns.set(font="Malgun Gothic")
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

        if not hasattr(self.koges, "data_multiclass"):
            raise ValueError("multiclass dataset이 정의되지 않았습니다.")

        self.results = []
        y = self.koges.y[0]
        _kg = KogesData.copy(self.koges)
        for i, scaler in enumerate(self.scalers):
            _kg.data = _kg.data_multiclass
            _kg.data[y] = _kg.data[y].astype(int)

            _kg = modelclass.__scale(_kg, scaler=scaler)
            X_train, X_test, y_train, y_test = modelclass.__split(_kg)

            cnt = 0
            _W = np.random.randn(X_train.shape[1], _kg.n_class)
            if printCost:
                print(scaler.__name__)
            while cnt < max_iter:
                z = np.dot(X_train, _W)

                predictions = z - (z.max(axis=1).reshape([-1, 1]))
                softmax = np.exp(predictions)
                softmax /= softmax.sum(axis=1).reshape([-1, 1])

                sample_size = y_train.shape[0]
                cost = -np.log(softmax[np.arange(len(softmax)), y_train]).sum()
                cost /= sample_size
                cost += (1e-5 * (_W**2).sum()) / 2

                if printCost and cnt % 10000 == 0:
                    print(f"iter: {cnt} cost: {cost}")
                if cost < threshold:
                    break

                softmax[np.arange(len(softmax)), y_train] -= 1
                grad = np.dot(X_train.transpose(), softmax) / sample_size
                grad += 1e-5 * _W

                _W -= learning_rate * grad
                cnt += 1

            prediction = np.dot(X_test, _W)
            self.results.append(
                softmaxResult(scaler=scaler, y_test=y_test, prediction=prediction)
            )

        if not display_best:
            ncol = len(self.scalers)
            fig, ax = plt.subplots(
                nrows=1,
                ncols=ncol,
                figsize=(ncol * 4, 4),
                constrained_layout=True,
                sharey=False,
            )
            for i, r in enumerate(self.results):
                plt.subplot(1, ncol, i + 1)
                plt.title(f"{r.scaler.__name__} (accuracy={r.accuracy:.2f})")
                self.__muticlassRoc(result=r)

            fig.supxlabel("False Positive Rate")
            fig.supylabel("True Positive Rate")
            fig.suptitle("Multiclass ROC curve")
            if display_roc_curve:
                plt.show()

            fig2, ax = plt.subplots(
                nrows=1,
                ncols=ncol,
                figsize=(ncol * 4, 4),
                constrained_layout=True,
                sharey=True,
            )
            for i, r in enumerate(self.results):
                plt.subplot(1, ncol, i + 1)
                sns.heatmap(
                    r.conf_matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=self.koges.classes,
                    yticklabels=self.koges.classes,
                )
                plt.title(f"Softmax ({r.scaler.__name__})")
            fig2.suptitle("Confusion matrix")
            if display_confusion_matrix:
                plt.show()
            self.koges.SAVE["multiclassRoc"] = fig
            self.koges.SAVE["softmaxClassifier"] = fig2
        else:
            r = self.__get_best()
            fig, ax = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(9, 4),
                constrained_layout=True,
                sharey=False,
            )
            # plt.title(f"{r.scaler.__name__} (accuracy={r.accuracy:.2f})")
            plt.subplot(1, 2, 1)
            self.__muticlassRoc(result=r)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")

            plt.subplot(1, 2, 2)
            sns.heatmap(
                r.conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.koges.classes,
                yticklabels=self.koges.classes,
            )
            fig.suptitle(f"Softmax ({scaler.__name__})")

            if display_roc_curve or display_confusion_matrix:
                plt.show()
            self.koges.SAVE["softmaxClassifier"] = fig

    def equation(
        self,
        isdisplay=True,
    ):
        from pykoges.utils import isdiscrete, name_map
        from sklearn.linear_model import LinearRegression
        from IPython.display import display, Math

        # LaTeX 형식의 모델 식 생성
        if isdiscrete(self.koges.q, self.koges.y[0]):
            return
        if not hasattr(self, "coef") or not hasattr(self, "bias"):
            return
        b = "{:.2f}".format(self.bias)
        W = ["{:.2f}".format(x) for x in self.coef]
        lines = []
        X = [name_map.get(x, x) for x in self.koges.x]
        for w, x in sorted(zip(W, X), reverse=True):
            if float(w) >= 0:
                w = "+ " + w
            lines.append(f"{w} \\times \\text{{{x}}}")

        y = self.koges.y[0]
        y = name_map.get(y, y)
        line = "".join(lines)
        if isinstance(self.model, LinearRegression):
            equation = f""" y({y}) = {b} {line}"""
        else:
            equation = f"X = {b} {line}\n"
            equation += f"P(abnormal, {y}) = P(y=1) = \\frac{{1}}{{1 + e^{{-X}}}}"
        if isdisplay:
            display(Math(equation))
        self.koges.SAVE["equation"] = equation
