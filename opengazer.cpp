#define BOOST_FILESYSTEM_VERSION 3

//#define EXPERIMENT_MODE
//#define DEBUG

#include "MainGazeTracker.h"
#include <QApplication>

int main(int argc, char **argv) {
    QApplication app(argc, argv);

    try {
        MainGazeTracker tracker(argc, argv);

        return app.exec();
    }
    catch (Utils::QuitNow) {
        app.quit();
    }

    return 0;
}
