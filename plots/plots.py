import matplotlib.pyplot as plt


def vector_to_image(width, size, *args):
    n = len(args)
    fig = plt.figure()
    for i, args in enumerate(args):
        plt.subplot(1, n, i + 1)
        plt.imshow(args.reshape(width, width), interpolation='None', cmap=plt.get_cmap('gray'))
        plt.axis('off')
    fig.tight_layout(pad=0)
    fig.set_size_inches(w=n * size, h=size)
    plt.show()


def scatter_plot(data, class_labels, positive_class_label, negative_class_label):
    # x = data['feature_vectors']['feature0']
    # y = data['feature_vectors']['feature1']
    x = data[:, 0]
    y = data[:, 1]
    # class_labels = data['class_labels']
    positive_color = 'red'
    negative_color = 'blue'
    colors = [positive_color if cl == positive_class_label else negative_color for cl in class_labels]

    plt.scatter(x, y, s=5, c=colors, alpha=0.5)
    plt.show()
