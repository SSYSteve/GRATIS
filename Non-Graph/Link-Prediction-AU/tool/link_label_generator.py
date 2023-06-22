import numpy as np

def save_as_npy(link_label_file, new_link_label_file, num_classes=12):
    link_label_list = np.loadtxt(link_label_file)
    
    link_label_list = [i.reshape(num_classes*num_classes,4) for i in link_label_list]
    link_label_list = np.array(link_label_list)

    indices = []

    for i in range(num_classes):
        for j in range(i+1, num_classes):
            indices.append(num_classes*i + j)

    link_label_list = link_label_list[:, indices, :]
    link_label_list = np.argmax(link_label_list, axis=2)

    np.save(new_link_label_file, link_label_list)

    return


def generate_co_occurrence_labels(label_file, link_label_file):
    with open(label_file, 'rb') as f:
        data = np.loadtxt(f)

    link_labels = open(link_label_file, 'x+')
    link_labels.close()
    link_labels = open(link_label_file, 'a+')

    for img in range(len(data)):
        for first_au in range(len(data[img])):
            for second_au in range(len(data[img])):
                if data[img][first_au] and data[img][second_au]:
                    # both active
                    link_labels.write("1 0 0 0 ")

                elif data[img][first_au] and not data[img][second_au]:
                    # only the first one active
                    link_labels.write("0 1 0 0 ")

                elif not data[img][first_au] and data[img][second_au]:
                    # only the second one active
                    link_labels.write("0 0 1 0 ")

                else:
                    # neither active
                    link_labels.write("0 0 0 1 ")

        link_labels.write("\n")

    link_labels.close()

    return


def main():
    list_dir = "data/DISFA/list/"
    for fold in range(3):
        
        generate_co_occurrence_labels(list_dir + "DISFA_test_label_fold"+str(fold+1)+".txt", list_dir + "DISFA_test_co-occ_label_fold"+str(fold+1)+".txt")
        save_as_npy(list_dir + "DISFA_test_co-occ_label_fold"+str(fold+1)+".txt", list_dir + "DISFA_test_co-occ_label_fold"+str(fold+1)+".npy", num_classes=8)

        generate_co_occurrence_labels(list_dir + "DISFA_train_label_fold"+str(fold+1)+".txt", list_dir + "DISFA_train_co-occ_label_fold"+str(fold+1)+".txt")
        save_as_npy(list_dir + "DISFA_train_co-occ_label_fold"+str(fold+1)+".txt", list_dir + "DISFA_train_co-occ_label_fold"+str(fold+1)+".npy", num_classes=8)

#-----------------------------------------------------#
if __name__ == "__main__":
    main()

