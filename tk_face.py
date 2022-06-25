"""
功能：GUI界面设计
时间：2022年5月16日16:37:20
作者：陈子含
"""
import os, shutil, sys
import tkinter as tk
from PIL import Image, ImageTk
import cv2 as cv
from utils import SCALE, IMAGE_SIZE, putText_ch, \
    MAX_NUM, face_train, THRESHOLD, face_cascade, \
    p_path, new_id
import joblib, time
import face_recognition as face
from tkinter import messagebox
from bidict import bidict
from glob import glob
import numpy as np


"窗口"
win = tk.Tk()
win.title('orczh')
win.geometry('800x480')
win.wm_resizable(0, 0)
win.iconbitmap(p_path + '/data/01.ico')
"窗口全局变量"
win.num = 0
win.mode = 0
win.start = False
win.time = time.time()


class FaceModel:
    """辅助变量储存"""
    def __init__(self):
        if not os.path.exists(p_path + '/model/'):
            os.mkdir(p_path + '/model/')
        if not os.path.exists('face_data/'):
            os.mkdir(p_path + '/face_data/')
        # 加载已有模型
        if os.listdir(p_path + '/model/'):
            model_name = os.listdir(p_path + '/model/')[-1]
            self.knn = joblib.load(p_path + f'/model/{model_name}')
            self.people = self.knn.people
        else:
            # 无模型时重新训练
            names = os.listdir(p_path + '/face_data/')
            # 删除空人脸数据
            for name in names:
                if not os.listdir(p_path + f'/face_data/{name}/figure'):
                    shutil.rmtree(p_path + f'/face_data/{name}')
                    names.remove(name)
            if len(names) == 0:
                self.people = bidict()
                messagebox.showinfo('提示', '当前人脸数据库为空！\n请新增人脸！')
                return
            else:
                self.people = bidict(zip(names, range(len(names))))
                self.knn = face_train(self.people)

    def retrain(self):
        names = os.listdir(p_path + '/face_data/')
        del_p = [p for p in names if p not in self.people.keys()]
        ask = 'yes'
        if del_p:
            ask = messagebox.askquestion('提示', '由于数据缺失，下列人脸信息将丢失\n'+'\n'.join(del_p)+'\n还要继续吗？')
        if ask == 'yes':
            if names:
                people = bidict(zip(names, range(len(names))))
                self.knn = face_train(people)
                self.people = self.knn.people
                messagebox.showinfo('提示', '重新训练成功！')
            else:
                messagebox.showinfo('提示', '重新训练失败！\n当前人脸数据库为空，请新增人脸！')

    def train_new_face(self, name):
        label = new_id(self.people)
        path = p_path + f'/face_data/{name}/figure/*'
        fig_paths = glob(path)
        if not fig_paths:
            shutil.rmtree(p_path + f'/face_data/{p}/')
        cods = []
        for fig_path in fig_paths:
            print(f'\r{fig_path}', end='')
            fig = Image.open(fig_path)
            fig = np.asarray(fig)[:, :, ::-1]
            faces = face_cascade.detectMultiScale(fig)
            locs = list(map(lambda x: (x[1], x[0] + x[2], x[1] + x[3], x[0]), faces))
            cods += (face.face_encodings(fig, locs, model='small'))
        print('')
        if len(cods) > 0:
            temp = np.stack(cods)
            np.savetxt(p_path + f'/face_data/{name}/codes.txt', temp)
        labels = [label] * len(cods)
        data = list(zip(cods, labels))
        X, Y = zip(*data)
        try:
            self.knn.fit(X, Y)
            self.people[name] = label
            self.knn.people = self.people
            joblib.dump(self.knn, p_path + f'/model/{self.knn.name}')
            return True
        except:
            return False

fm = FaceModel()


def face_normal(frame):
    """
    win.mode = 0
    把图像处理后直接输出到窗口
    """
    frame = cv.flip(frame, 1)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.putText(frame, 'FPS:{:d}'.format(int(1 / (time.time() - win.time))), (0, 20), 0, 0.5, (255, 0, 0), 1)
    win.time = time.time()
    return frame


def face_saving(frame):
    """
    win.mode = 1
    新增人脸信息，读取并储存图像中的人脸信息
    （人脸需要靠近摄像头保证准确率）
    结束后重新进行训练，并更新模型
    """
    frame = cv.flip(frame, 1)
    faces = list(face_cascade.detectMultiScale(frame))
    HEIGHT, WIDTH = frame.shape[:2]
    if faces:
        (x, y, w, h) = faces[np.argmax(list(map(lambda x:x[2] * x[3], faces)))]
        if w * h / (WIDTH * HEIGHT) > 0.2:
            win.num += 1
            rect = (int(x - w * (SCALE - 1) / 2), int(y - h * (SCALE - 1) / 2),
                    int(x + w * (SCALE + 1) / 2), int(y + h * (SCALE + 1) / 2))
            fig = frame[rect[1]:rect[3], rect[0]:rect[2]]
            if fig.size != 0:
                fig = cv.resize(fig, IMAGE_SIZE)
                fig = Image.fromarray(fig[:, :, ::-1])
                fig.save(p_path + f'/face_data/{new_name.get()}/figure/' + f'{win.num}.jpg')
                fig.close()
                frame = cv.rectangle(frame, rect[:2], rect[2:], (0, 255, 0), 1)
        else:
            frame = cv.rectangle(frame, (245, 352), (395, 403), (255, 255, 255), -1)
            frame = putText_ch(frame, '请靠近一些', (260, 360), text_size=24, color=(49, 50, 255))
    else:
        frame = cv.rectangle(frame, (245, 352), (395, 403), (255, 255, 255), -1)
        frame = putText_ch(frame, '未识别人脸', (260, 360), text_size=24, color=(49, 50, 255))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.putText(frame, 'FPS:{:d}'.format(int(1 / (time.time() - win.time))), (0, 20), 0, 0.5, (255, 0, 0), 1)

    if win.num >= MAX_NUM:
        messagebox.showinfo("提示", f'{new_name.get()}人脸数据保存成功！\n开始训练')
        win.mode = 0
        win.num = 0
        ret = fm.train_new_face(new_name.get())
        if ret:
            messagebox.showinfo("提示", '训练完成！')
            bt22.delete(0, bt22.size() - 1)
            for p, l in fm.people.items():
                bt22.insert(l, p)
        else:
            messagebox.showwarning("提示", '训练失败！')
        bt4change()
    win.time = time.time()
    return frame


def face_predict(frame):
    """
    win.mode = 2
    调用人脸识别模型，开始人脸识别，用方框标注人脸，并标出姓名
    """
    frame = cv.flip(frame, 1)
    faces = face_cascade.detectMultiScale(frame)
    if len(faces) > 0:
        locs = list(map(lambda x: (x[1], x[0] + x[2], x[1] + x[3], x[0]), faces))
        encodings = face.face_encodings(frame, locs, model='small')
        proba = fm.knn.predict_proba(encodings)
        for i in range(len(faces)):
            (x, y, w, h) = faces[i]
            rect = (int(x - w * (SCALE - 1) / 2), int(y - h * (SCALE - 1) / 2),
                    int(x + w * (SCALE + 1) / 2), int(y + h * (SCALE + 1) / 2))
            if proba[i].max() > THRESHOLD[0]:
                idx = proba[i].argmax()
                name = fm.people.inverse.get(idx, None)
                if name:
                    frame = cv.rectangle(frame, rect[:2], rect[2:], (0, 255, 0), 1)
                    frame = putText_ch(frame, name, rect[:2], text_size=16)
            # elif proba[i].max() > THRESHOLD[1]:
            #     frame = cv.rectangle(frame, rect[:2], rect[2:], (0, 255, 0), 1)
            #     frame = putText_ch(frame, '未知', rect[:2], text_size=16)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.putText(frame, 'FPS:{:d}'.format(int(1 / (time.time() - win.time))), (0, 20), 0, 0.5, (255, 0, 0), 1)
    win.time = time.time()
    return frame


new_name = tk.StringVar()


def add_face(event):
    """
    ‘新增’的回调函数
    """
    if new_name.get() != '':
        bt4['text'] = '取消人脸采集'
        bt4['width'] = 15
        bt4['height'] = 1
        path = p_path + f'/face_data/{new_name.get()}/figure'
        if os.path.exists(path):
            shutil.rmtree(p_path + f'/face_data/{new_name.get()}')
        os.makedirs(path)
        win.mode = 1
    else:
        messagebox.showwarning('提示', '输入为空，请检查！')


def start():
    """
    ‘开始’的回调函数
    """
    if win.mode == 0:
        if fm.people == {}:
            messagebox.showwarning('警告', '当前人脸数据库为空！\n请新增人脸！')
        else:
            bt3['text'] = '结束'
            win.mode = 2
    elif win.mode == 2:
        bt3['text'] = '开始'
        win.mode = 0


def delete(event):
    """
    选中姓名按<Delete>删除该信息并重新训练
    """
    if event.keycode == 46:
        try:
            val = bt22.get(bt22.curselection())
            ask = messagebox.askquestion('警告', f'删除后将会从现有数据库重新训练\n确定要删除{val}的人脸信息吗？', icon='warning')
            if ask == 'yes':
                if os.path.exists(f'/face_data/{val}'):
                    shutil.rmtree(p_path + f'/face_data/{val}')
                    messagebox.showinfo('提示', f'{val}信息已删除！')
                bt22.delete(bt22.curselection())
                bt22.focus_displayof()
                fm.retrain()
        except:
            messagebox.showwarning('提示', '没有选择任何条目！')
    else:
        messagebox.showinfo('提示', '按下Delete删除该信息！')


def view_data(event):
    """
    双击姓名查看数据库
    """
    try:
        val = bt22.get(bt22.curselection())
        if os.path.exists(p_path + f'/face_data/{val}/figure'):
            os.startfile(p_path + f'/face_data/{val}/figure')
        else:
            messagebox.showinfo('提示', f'{val}人脸数据不存在！')
    except:
        messagebox.showinfo('提示', f'没有选择任何条目！')


def bt4change():
    if bt4['text'] == 'exit':
        bt4['text'] = '取消人脸采集'
        bt4['width'] = 15
        bt4['height'] = 1
    else:
        bt4['text'] = 'exit'
        bt4['width'] = 7
        bt4['height'] = 1
    new_name.set('')


def exittk():
    """
    退出程序
    """
    if win.mode == 1:
        win.mode = 0
        win.num = 0
        shutil.rmtree(p_path + f'/face_data/{new_name.get()}')
        bt4change()
    else:
        # win.quit()
        sys.exit(0)

"窗口布局"
f = tk.Frame(win)
f.grid(row=0, column=1, sticky='nw')
# 视频窗口
can = tk.Label(win, width=640, height=480)
can.grid(row=0, column=0, sticky='nw')
# 新增
bt11 = tk.Label(f, text='新增', width=10)
bt11.grid(row=0, column=0, columnspan=1, sticky='e', ipady=5, pady=10)
bt12 = tk.Entry(f, width=10)
bt12.grid(row=0, column=1, columnspan=1, sticky='e', ipady=10, pady=10)
bt12.bind('<Return>', add_face)
bt12['textvariable'] = new_name
# 重新训练
bt13 = tk.Button(f, text='重新训练', width=10, command=fm.retrain)
bt13.grid(row=1, column=0, columnspan=2, ipady=5, pady=5)
# 数据库
bt21 = tk.Label(f, text='数据库', width=10)
bt21.grid(row=2, column=0, columnspan=2, ipady=10)
bt22 = tk.Listbox(f, height=8)
bt22.grid(row=3, column=0, columnspan=2, rowspan=4, ipady=5, sticky='nswe')
for p, l in fm.people.items():
    bt22.insert(l, p)
bt22.bind("<Delete>", delete)
bt22.bind("<Double-Button-1>", view_data)
# 开始
bt3 = tk.Button(f, text='开始', width=10, command=start)
bt3.grid(row=7, column=0, columnspan=2, ipady=10, pady=10)
# 退出
bt4 = tk.Button(f, text='exit', width=7, height=1, fg='#FF0000', font=('simsun', 12), pady=1, command=exittk)
bt4.grid(row=8, column=0, columnspan=2, ipady=10, pady=10)


cap = cv.VideoCapture(0)


def get_mode():
    if win.mode == 0:
        return face_normal
    elif win.mode == 1:
        return face_saving
    elif win.mode == 2:
        return face_predict


def video_stream():
    mode = get_mode()
    ret, frame = cap.read()
    frame = mode(frame)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    can.imgtk = imgtk
    can.configure(image=imgtk)
    can.after(1, video_stream)

video_stream()
win.mainloop()
