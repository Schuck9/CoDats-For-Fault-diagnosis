
def load_file(self, filename):
    """ Load ZIP file containing all the .txt files """
    with zipfile.ZipFile(filename, "r") as archive:
        train_data, train_labels, train_subjects = self.get_data(archive, "train")
        test_data, test_labels, test_subjects = self.get_data(archive, "test")

    all_data = np.vstack([train_data, test_data]).astype(np.float32)
    all_labels = np.hstack([train_labels, test_labels]).astype(np.float32)
    all_subjects = np.hstack([train_subjects, test_subjects]).astype(np.float32)

    # All data if no selection
    if self.users is None:
        return all_data, all_labels

    # Otherwise, select based on the desired users
    data = []
    labels = []

    for user in self.users:
        selection = all_subjects == user
        data.append(all_data[selection])
        current_labels = all_labels[selection]
        labels.append(current_labels)

    x = np.vstack(data).astype(np.float32)
    y = np.hstack(labels).astype(np.float32)

    # print("Selected data:", self.users)
    # print(x.shape, y.shape)

    return x, y