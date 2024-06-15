import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QRect
from PyQt5 import QtWidgets

class RedRectangleOverlay(QWidget):
    def __init__(self, width, height, parent=None):
        super(RedRectangleOverlay, self).__init__(parent)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(width, height)  # Use custom size

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRect(0, 0, self.width(), self.height())
        painter.setBrush(QBrush(Qt.red))
        painter.setPen(QPen(Qt.red))
        # Draw the rectangle
        painter.drawRect(rect)
    
    def mousePressEvent(self, event):
        QtWidgets.qApp.quit()

    def move_to_left(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        x = 0
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def move_to_right(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        x = screen_geometry.width() - self.width()
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
    
    def move_to_top(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = 0
        self.move(x, y)

    def move_to_bottom(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = screen_geometry.height() - self.height()
        self.move(x, y)

def open_overlay():
    app = QApplication(sys.argv)
 
    screen_geometry = QApplication.desktop().screenGeometry()
    width_inline=int(screen_geometry.width()*0.008)
    width_block=int(screen_geometry.width())
    height_inline=int(screen_geometry.height())
    height_block=int(screen_geometry.height()*0.01)

    # Create and position the left overlay with custom size
    left_overlay = RedRectangleOverlay(width_inline, height_inline)
    left_overlay.move_to_left()
    left_overlay.show()

    # Create and position the right overlay with custom size
    right_overlay = RedRectangleOverlay(width_inline, height_inline)
    right_overlay.move_to_right()
    right_overlay.show()

    # Create and position the top overlay with custom size
    top_overlay = RedRectangleOverlay(width_block, height_block)
    top_overlay.move_to_top()
    top_overlay.show()

    # Create and position the bottom overlay with custom size
    bottom_overlay = RedRectangleOverlay(width_block, height_block)
    bottom_overlay.move_to_bottom()
    bottom_overlay.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    open_overlay()


