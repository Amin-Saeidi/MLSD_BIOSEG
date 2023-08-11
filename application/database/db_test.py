import database
import time

if __name__ == "__main__":
    db = database.DB()
    lr = database.Image_record(image="Google.com", class_name="Website")
    db.add_image(record=lr)
    time.sleep(1)
    for row in db.get_image():
        print (row.image)