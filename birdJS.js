const video = document.getElementById('myVideo');
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');

let model;
async function setupApp() {
    console.log("1. טוען מודל...");
    try {
        model = await tf.loadGraphModel('yolov8n_web_model/model.json');
        console.log("2. המודל נטען בהצלחה!");
    } catch (e) {
        console.error("❌ שגיאה בטעינת המודל:", e);
        return;
    }

    // פונקציה שתתחיל את הלולאה
    const startLoop = () => {
        console.log("4. מתחיל לולאת זיהוי (detectFrame)...");
        detectFrame();
    };

    console.log("3. בודק מצב וידאו. ReadyState:", video.readyState);

    // אם הוידאו כבר מנגן או מוכן
    if (video.readyState >= 2) {
        startLoop();
    } else {
        // מחכה שהוידאו יתחיל לנגן פיזית
        video.addEventListener('play', startLoop);
    }
}

async function detectFrame() {
    console.log("--- פריים חדש ---"); // זה אמור להציף את הלוג אם זה עובד

    if (video.paused || video.ended) {
        console.log("הוידאו בהפסקה או הסתיים");
        return;
    }

    const input = tf.tidy(() => {
        return tf.browser.fromPixels(video)
            .resizeBilinear([320, 320])
            .toFloat()
            .div(255.0)
            .expandDims(0);
    });

    try {
        const res = model.execute(input);
        console.log("Output Shape:", res.shape); // בדיקת מבנה הפלט

        // כאן הקוד שכתבנו קודם לבדיקת הציונים...
        const { values } = tf.topk(tf.tidy(() => res.squeeze().transpose().slice([0, 4], [-1, 1]).squeeze()), 5);
        console.log("Top 5 Scores:", values.arraySync());

        res.dispose();
    } catch (err) {
        console.error("❌ שגיאה בזמן הרצת המודל:", err);
    }

    input.dispose();
    requestAnimationFrame(detectFrame);
}

setupApp();