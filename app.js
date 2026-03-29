const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const statusText = document.getElementById('status');
const ctx = canvas.getContext('2d');

let model;
let frameCount = 0;

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
    
    frameCount++;

    // 1. מריצים את המודל ומעדכנים את הריבועים רק פעם ב-3 פריימים
    if (frameCount % 3 === 0) {
        
        const detections = tf.tidy(() => {
            // א. עיבוד מקדים (Pre-processing)
            let img = tf.browser.fromPixels(video);
            
            // במידה והתוצאות הפוכות/לא טובות, בטל את ה-comment מהשורה הבאה:
            // img = img.reverse(2); 
            
            const input = img.resizeBilinear([IMGSZ, IMGSZ])
                             .toFloat()
                             .div(255.0)
                             .expandDims(0);

            // ב. הרצת המודל (res נוצר כאן בתוך הבלוק!)
            const res = model.execute(input);

            // ג. עיבוד תוצאות (Post-processing)
            // ניגש לאינדקסים של res רק עכשיו כשהוא בטוח נוצר
            const rawBoxes = res[0];
            const rawLogits = res[1];

            // המרה לסיגמואיד
            const rawScores = tf.sigmoid(rawLogits);

            // שולחים החוצה מ-tidy רק מערכי JS פשוטים (Numbers)
            return {
                boxes: rawBoxes.squeeze().arraySync(),
                scores: rawScores.squeeze().arraySync(),
                classIds: new Array(rawScores.size).fill(0)
            };
        });

        // 2. ציור התיבות על הקנבס (רק בפריים שבו התבצע זיהוי)
        drawBoxes(detections);
    }

    // 3. בקשה לפריים הבא - קורה תמיד כדי שהוידאו ימשיך לרוץ חלק
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
