from methods import get_order, train_model
# get list
overall_list, _ = get_order()
# set i to the number of mosaics -1
num = 0
i = 21
# start model training with number of repetitions
# set all parameters
for num in range(5):
    print("On repetition ", num)
    train_model(train_path=overall_list[i], extra_path=overall_list[24],
                val_path="/home/s6nocrem/project/croptypes/csv/mosaics_validation/mosaics_validation_ssh.csv",
                dataset_size=len(overall_list[i]), dataset_number=i+1, repetition=num, net="VGG", num_classes=5, batch_size=100, epochs=10, mode='other')
