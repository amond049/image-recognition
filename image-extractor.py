# Hopefully this will be a script that will load the necessary images to where they are needed
import os
import shutil

current_working_dir = os.getcwd()
image_parent_folder = current_working_dir + "\\stanford_cars_type"

class_folders = os.listdir(image_parent_folder)

for folder in class_folders:
    folder_path = os.path.join(image_parent_folder, folder)
    contents = os.listdir(folder_path)

    number_of_images = len(contents)

    test_images = round(number_of_images * 0.2)
    training_images = round(number_of_images * 0.8 * 0.8)
    validation_images = number_of_images - test_images - training_images


    for i in range(test_images):
        shutil.copy(os.path.join(folder_path, contents[i]), os.path.join(current_working_dir + "\\body-type-dataset\\test\\" + folder, contents[i]))
    print("===========================")
    for i in range(test_images, training_images + test_images):
        shutil.copy(os.path.join(folder_path, contents[i]), os.path.join(current_working_dir + "\\body-type-dataset\\train\\" + folder, contents[i]))
    print("===========================")
    for i in range(test_images + training_images, validation_images + test_images + training_images):
        shutil.copy(os.path.join(folder_path, contents[i]), os.path.join(current_working_dir + "\\body-type-dataset\\validation\\" + folder, contents[i]))
    print("===========================")


