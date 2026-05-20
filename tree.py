from sklearn import tree
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import cross_val_score
import pandas as pd


def plot_tree(clf, feature_names=None, class_names=None,
              title=None, filled=True, figsize=(10, 5), **kwargs):
    if feature_names is None:
        feature_names = clf.feature_names_in_.astype(str)
    if class_names is None:
        class_names = clf.classes_.astype(str)

    fig, ax = plt.subplots(figsize=figsize)
    tree.plot_tree(clf, feature_names=feature_names, class_names=class_names,
                   filled=filled, ax=ax, **kwargs);
    plt.suptitle(title, y=1)
    plt.tight_layout()
    plt.show()
    return fig, ax


def cost_complexity_pruning(base_clf, X_train, y_train,
                            cv=10, scoring='accuracy', skip_one_node=False,
                            plot_complexity=True, plot_cv_scores=True, cv_tol=1e-4,
                            figsize=(12, 5), num_cv_labels=10):

    path = base_clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    clf_dict = defaultdict(dict)
    clf_dict['ccp_alpha'] = ccp_alphas

    for ccp_alpha in ccp_alphas:
        if skip_one_node & (ccp_alpha == ccp_alphas[-1]):
            # Skip the trivial tree
            continue
        clf = tree.DecisionTreeClassifier(**{**base_clf.get_params(), 'ccp_alpha': ccp_alpha})
        clf_dict['cv_scores'][ccp_alpha] = cross_val_score(clf, X_train, y_train, cv=cv, scoring=scoring)
        clf.fit(X_train, y_train)
        clf_dict['clf'][ccp_alpha] = clf
        clf_dict['node_counts'][ccp_alpha] = clf.tree_.node_count
        clf_dict['max_depth'][ccp_alpha] = clf.tree_.max_depth

    if plot_complexity:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        lines = []
        labels = []

        for stat, ax, color in zip(['node_counts', 'max_depth'], [ax1, ax2], ['b', 'r']):
            s = pd.Series(clf_dict[stat])
            s.plot(style='.', ax=ax, color=color)
            s.plot(label=stat, drawstyle="steps-post", marker="o", color=color, ax=ax)

            ax_lines, ax_labels = ax.get_legend_handles_labels()
            lines.append(ax_lines)
            labels.append(ax_labels)
            ax.set_ylabel(stat)

        ax1.set_title("Tree Complexity")
        ax1.set_xlabel("alpha")
        ax2.legend(lines[0] + lines[1], labels[0] + labels[1], loc=0)
        plt.show()

    if plot_cv_scores:
        cv_scores = pd.DataFrame(clf_dict['cv_scores'])
        cv_scores.plot(kind='box', figsize=figsize, showmeans=True,
                       title=f"{cv}-fold Cross Validation Scores with scoring={scoring}")

        step = max(len(cv_scores.columns) // num_cv_labels, 1)
        plt.xticks(ticks=range(1, len(cv_scores.columns) + 1, step),
                   labels=cv_scores.columns.map(lambda s: f"{s:.4f}")[::step]);
        plt.xlabel("alpha")
        plt.show()

    cv_mean = pd.DataFrame(clf_dict['cv_scores']).mean()
    best_cv = cv_mean.max()
    best_alpha = cv_mean.ge(best_cv - cv_tol).idxmax()

    best_clf = clf_dict['clf'][best_alpha]
    title = rf"Pruned at $\alpha$={best_alpha:.4f}, CV {scoring} Score= {cv_mean[best_alpha]:.4f}"
    plot_tree(best_clf, title=title)
    return best_alpha, clf_dict