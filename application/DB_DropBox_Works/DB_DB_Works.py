import dropbox
import string
import random
import sys
sys.path.append("..")
import database.database as database


class DB_DropBox_Works:

    def __init__(self):
        self.dropbox_access_token = "sl.Bj8lj711SbojG7icwllQmujhyB0lfrKhUNtghKnloxtUzQ599DcnptU1lulH7JZFZGnZs_2HhLk0gYqDERP5_u3wpHJrks8eFgFMoHWC-avia213xT8TtOvttirpR5XPpig4FMurjUqM70pa36cJdEc"

    def random_name(self, l = 7):
        res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=l))
        # print("The generated random string : " + str(res))
        return res


    def connecting_dropbox(self):
        client = dropbox.Dropbox(self.dropbox_access_token)
        # print("[SUCCESS] dropbox account linked")
        return client

    def upload_image(self, client, image_path):
        
        dropbox_path = f"/mlsd-bioseg-photos/{image_path}"

        client.files_upload(open(image_path, "rb").read(), dropbox_path)
        # print("[UPLOADED] {}".format(image_path))
        
        return dropbox_path

    def database_add(self, db, client, image_path = "D:/Downloads/Chrome/0.png", pr = "CM", gt = "CM"):

        image_dropbox_path = self.upload_image(client, image_path)        
        lr = database.Image_record(image=image_dropbox_path, class_name_predicted=pr, class_name_gt=gt)
        db.add_image(record=lr)


if __name__ == "__main__":
    db = database.DB()        
    for row in db.get_image():
        print (f"id: {row.id} - image_dropbox_link: {row.image} - image_predicted_clsss: {row.class_name_predicted} - image_gt_class: {row.class_name_gt}" )
            
