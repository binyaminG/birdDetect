const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const statusText = document.getElementById('status');
const ctx = canvas.getContext('2d');

let model;

// הגדרות המודל - וודא שהן תואמות לייצוא מה-Colab
const IMGSZ = 320;
const CONF_THRESHOLD = 0.45; // סף רגישות (ניתן להוריד אם לא רואים כלום)

async function setupApp() {
    console.log("1. טוען מודל...");
    try {
        // טעינת המודל מהתיקייה
        model = await tf.loadGraphModel('best_web_model/model.json');
        console.log("2. המודל נטען בהצלחה!");
        statusText.innerText = "המודל מוכן, מתחיל בזיהוי...";

        // התחלת הלולאה כשהוידאו מוכן
        if (video.readyState >= 2) {
            startDetection();
        } else {
            video.onloadedmetadata = () => startDetection();
        }
    } catch (e) {
        console.error("❌ שגיאה בטעינה:", e);
        statusText.innerText = "שגיאה בטעינה: " + e.message;
    }
}

function startDetection() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    detectFrame();
}

async function detectFrame() {
    if (video.paused || video.ended) return;

    // 1. עיבוד מקדים (Pre-processing)
    const input = tf.tidy(() => {
        let img = tf.browser.fromPixels(video);

        // אם התוצאות עדיין לא טובות, נסה להוסיף את השורה הבאה:
        // img = img.reverse(2); // היפוך RGB ל-BGR

        return img.resizeBilinear([IMGSZ, IMGSZ])
                  .toFloat()
                  .div(255.0)
                  .expandDims(0);
    });


    // 2. הרצת המודל
    const res = model.execute(input);

    // 3. עיבוד תוצאות (Post-processing)
    const detections = tf.tidy(() => {
        // מכיוון ש-res הוא מערך, ניגש לאיברים שלו לפי אינדקס
        // אינדקס 0 מכיל את התיבות (Boxes), אינדקס 1 מכיל את הציונים (Scores)
        const rawBoxes = res[0];
        const rawLogits = res[1];

        // המרה לסיגמואיד כדי לטפל במספרים השליליים שראית
        const rawScores = tf.sigmoid(rawLogits);

        return {
            // כאן אנחנו מפעילים את ה-squeeze על הטנזורים הספציפיים
            boxes: rawBoxes.squeeze().arraySync(),
            scores: rawScores.squeeze().arraySync(),
            // במודל ראשים (קלאס יחיד), ה-ID הוא תמיד 0
            classIds: new Array(rawScores.size).fill(0)
        };
    });

    drawBoxes(detections);

    input.dispose(); // האינפוט הוא טנזור בודד, זה תקין

    if (Array.isArray(res)) {
        res.forEach(t => t.dispose());
    } else {
        res.dispose();
    }

    requestAnimationFrame(detectFrame);
}

function drawBoxes(detections) {
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { boxes, scores } = detections;

    for (let i = 0; i < scores.length; i++) {
        if (scores[i] > CONF_THRESHOLD) {
            // קבלת הערכים המנורמלים (0-1) כפי שראינו בלוגים
            let [x_center, y_center, width, height] = boxes[i];

            // 1. הגבלת המרכז לתוך גבולות הפריים (0 עד 1)
            // זה מונע מהריבוע "לטפס" או "לרדת" מחוץ לוידאו
            const safeX = Math.max(0, Math.min(x_center, 1));
            const safeY = Math.max(0, Math.min(y_center, 1));

            // 2. חישוב מימדים סופיים בפיקסלים לפי גודל הקנבס האמיתי
            const w = width * canvas.width;
            const h = height * canvas.height;

            // 3. חישוב הפינה השמאלית העליונה (x1, y1)
            // אנחנו מחסירים חצי מהרוחב/גובה מהמרכז המנורמל מוכפל בקנבס
            const x1 = (safeX * canvas.width) - (w / 2);
            const y1 = (safeY * canvas.height) - (h / 2);

            // 4. ציור
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, w, h);

            // הוספת אחוז הביטחון מעל הריבוע
            ctx.fillStyle = "#00FF00";
            ctx.font = "bold 16px Arial";
            ctx.fillText(`${(scores[i] * 100).toFixed(0)}%`, x1, y1 > 20 ? y1 - 5 : 20);
        }
    }
}

setupApp();