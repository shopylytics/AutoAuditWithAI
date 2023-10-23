from ultralytics import YOLO
import pickle
import cv2
import math


def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

    pklmodelfilename = 'Pklmodel/Cd134C400n.pkl'

    model = pickle.load(open(pklmodelfilename,'rb'))


    #model=YOLO("YOLO-Weights/ppe.pt")
    #classNames = ['Protective Helmet', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots']
    classNames = ["dog", "person", "cat", "tv", "car", "meatballs", "marinara sauce", "tomato soup", "chicken noodle soup",
     "french onion soup", "chicken breast", "ribs", "pulled pork", "hamburger", "cavity", "Tata Tea Gold Window",
     "Tata Tea Gold", "Tata Sampann Chilli Powder", "Tata Sampann Window", "Tata Sampann Turmeric Powder",
     "Tata Sampann Coriander Powder", "MamyPoko Extra Absorb Pants", "MamyPoko Pants Standard Pant",
     "Pampers Premium Care Pants", "Pampers All round Protection Pants", "Mothercare Quick Absorb Diaper Pants",
     "Pampers New Baby Taped Diapers", "Pampers Active Baby Diapers", "Tata Chakra Gold", "Tata Chakra Gold Elaichi",
     "Tata Tea Gold Care", "Tata Tetley Green Tea", "Tata Tetley Green Lemon & Honey Tea", "Tata Tea Premium",
     "Brooke Bond Taj Mahal Tea", "Brooke Bond Red Label", "Brooke Bond Red Label Natural Care Tea",
     "Brooke Bond Red Label Tea", "Tata Agni Tea", "Tata Tea Elaichi Chai", "Tata Agni Strong Tea",
     "Tata Agni Adrak Tea", "Tata Tea Agni Elaichi", "Brook Bond 3 Roses", "Wagh Bakri Premium Spiced Tea",
     "SOCIETY TEA Masala", "Girnar Royal Cup Tea", "Society Tea", "Ganesh Premium Tea", "Brook Bond Taaza",
     "Tata Tea Kanan Devan Strong", "Tata Tea Kanan Devan Clasisic", "Tata Tea Gold Darjeeling",
     "Wagh Bakri Premium Leaf Tea", "Wagh Bakri Good Morning Premium Tea", "Wagh Bakri Gold", "Wagh Bakri Mili Premium",
     "Wagh Bakri Navchetan Leaf Tea", "Tata Tea Kanan Devan Golden Leaf", "Girnar Detox Green Tea",
     "Girnar Green Tea Cardamom", "Brooke Bond Taaza Tea Masala Chaska", "Tata Tea Gold Saffron", "Lipton Green Tea",
     "Girnar Masala Chai", "Girnar Cardamom Chai", "Tata Coffee Specials Hazelnut", "Nestle Gold Cappuccino",
     "Bru Green Label Coffee", "Bru Instant Coffee", "Tata Tea Premium Teaveda", "Tea Valley Classic Tea",
     "Tea Valley Royal", "Lipton Yellow Label Tea", "Nestea Lemon Ice Tea", "Marvel Red Tea", "Marvel Masala Tea",
     "Girnar Green Tea Mint", "Girnar Green Tea Lemon", "Girnar Green Tea Lemon & Honny", "Society Premium Green Tea",
     "Society Shake To Make Mango", "Wagh Bakri Instant Masala Tea", "Wagh Bakri Instant Elaichi", "AVT Premium Tea",
     "Everest Masala Tea", "Tata Tetley Lemon", "Wagh Bakri Instant Ginger Tea", "Macha Tea Classic",
     "Macha Tea Classic Tulsi Adrak Elaichi", "Organic India Classic Tulsi Green Tea",
     "Organic India Tulsi Detox Kahwa Green Tea", "Organic India Tulsi Sweet Rose",
     "Organic India Tulsi Ginger Turmeric Green Tea", "Society Pure Assam Dust Tea", "Nescafe Classic Instant Coffee",
     "Nescafe Classic Black Roast Coffee", "Nescafe Sunrise Instant Coffee", "Organic India Tulsi Honey Chamomile Tea","Organic India Tulsi Masala Chai","Dabur Vedic Tea","Tata Coffee Gold",
     "Nescafe Gold Blend Instant Coffee","Tata Coffee Grand Pouch","Tata Coffee Grand","Tetley Black Tea Lemon Twist",
     "Tetley Black Tea Masala Chai","Bru Gold Aromatic Instant Coffee","Bru Super Strong Instant Coffee","Nescafe Gold Cappuccino",
     "Nescafe Gold Choco Mocha","Tata Coffee Grand Filter Coffee","AVT Premium Coffee","Lipton Darjeeling Long Leaf Tea",
     "Tata Lal Ghora Tea","Girnar Variety Pack Chai","Girnar Ginger Chai","Girnar Express Chai","Girnar Stevia Chai",
     "Girnar 3 in 1 Coffee","Girnar Kashmiri Kahwa","Girnar Lemon Tea","Girnar Calming Green Tea","Girnar Haldi Doodh",
     "Girnar Safron Chai","Tata Tea Premium Kadak Assam Tea","Tata Tetley Elichi Tea","Nescafe 3 in 1 Coffee",
     "Tata Tetley Masala Tea","Tata Tetley Ginger Zing","Tata Tetley Ginger","Tata Tetley Ginger Tea","Tata Tetley Original","Tea Valley Gold"]

    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if class_name == 'Dust Mask':
                    color=(0, 204, 255)
                elif class_name == "Glove":
                    color = (222, 82, 175)
                elif class_name == "Protective Helmet":
                    color = (0, 149, 255)
                else:
                    color = (85,45,255)
                if conf>0.5:
                    cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
                    cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

        yield img
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
#cv2.destroyAllWindows()
