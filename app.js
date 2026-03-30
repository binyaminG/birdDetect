const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const statusText = document.getElementById('status');
const ctx = canvas.getContext('2d');

let model;
let frameCount = 0;
let lastGeminiCheck = 0;
const CHECK_INTERVAL = 60000; // דקה אחת במילישניות
let geminiOffset = { dx: 0, dy: 0 }; // ההזחה שנשמור

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

async function checkWithGemini(videoElement, yoloBox) {
    console.log("מבצע אימות מול Google AI...");
    
    // 1. חילוץ התמונה מהוידאו בפורמט Base64
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];

    // 2. בניית הפרומפט
    const prompt = `Identify the bounding box of the bird's HEAD in this image. 
    Return ONLY a JSON object in this format: {"ymin": percentage, "xmin": percentage, "ymax": percentage, "xmax": percentage}.
    The percentages should be integers between 0 and 1000 based on the image dimensions.`;

    try {
        // קריאה ל-Gemini (ודא שהגדרת את ה-API Key שלך קודם לכן)
        const response = await model.generateContent([
            prompt,
            { inlineData: { data: base64Image, mimeType: "image/jpeg" } }
        ]);
        
        const text = response.response.text();
        const geminiBox = JSON.parse(text.trim()); // חילוץ ה-JSON
        
        // 3. השוואה וחישוב ההזחה (Offset)
        // נניח ש-yoloBox מגיע בערכים מנורמלים 0-1000
        const yoloCenter = {
            x: (yoloBox.xmin + yoloBox.xmax) / 2,
            y: (yoloBox.ymin + yoloBox.ymax) / 2
        };
        const geminiCenter = {
            x: (geminiBox.xmin + geminiBox.xmax) / 2,
            y: (geminiBox.ymin + geminiBox.ymax) / 2
        };

        // חישוב ההפרש
        geminiOffset.dx = geminiCenter.x - yoloCenter.x;
        geminiOffset.dy = geminiCenter.y - yoloCenter.y;

        console.log(`הזחה מחושבת: X=${geminiOffset.dx.toFixed(2)}, Y=${geminiOffset.dy.toFixed(2)}`);
        
    } catch (error) {
        console.error("שגיאה בפנייה ל-Gemini:", error);
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

    // מריצים את המודל ומעדכנים את הריבועים רק פעם ב-3 פריימים
    if (frameCount % 3 === 0) {
        
        const detections = tf.tidy(() => {
            // 1. עיבוד מקדים
            let img = tf.browser.fromPixels(video);
            const input = img.resizeBilinear([IMGSZ, IMGSZ])
                             .toFloat()
                             .div(255.0)
                             .expandDims(0);

            // 2. הרצת המודל (כאן נוצר res)
            const res = model.execute(input);

            // 3. עיבוד תוצאות
            const rawBoxes = res[0];
            const rawLogits = res[1];
            const rawScores = tf.sigmoid(rawLogits);

            return {
                boxes: rawBoxes.squeeze().arraySync(),
                scores: rawScores.squeeze().arraySync(),
                classIds: new Array(rawScores.size).fill(0)
            };
        });
        console.log("Detections raw from model:", detections.boxes[0]);

        // בדיקת הגנה: מוודאים ש-geminiOffset מכיל מספרים תקינים
        const dx = (typeof geminiOffset.dx === 'number' && !isNaN(geminiOffset.dx)) ? geminiOffset.dx : 0;
        const dy = (typeof geminiOffset.dy === 'number' && !isNaN(geminiOffset.dy)) ? geminiOffset.dy : 0;
        
        // שיבוט הנתונים, הוספת ההזחה בבטחה ופירוק המערך
        const adjustedBoxes = detections.boxes.map(box => {
       // שליפת הנתונים מהמערך כפי שהמודל פולט
       let [x_center, y_center, width, height, confidence] = box;

    // הגנה מפני NaN (אם משהו השתבש, נחזיר 0)
    x_center = isNaN(x_center) ? 0 : x_center;
    y_center = isNaN(y_center) ? 0 : y_center;
    width = isNaN(width) ? 0 : width;
    height = isNaN(height) ? 0 : height;

    // הוספת ההזחה של גוגל למרכז התיבה (מחושב בבטחה)
    const dx = (typeof geminiOffset.dx === 'number' && !isNaN(geminiOffset.dx)) ? geminiOffset.dx : 0;
    const dy = (typeof geminiOffset.dy === 'number' && !isNaN(geminiOffset.dy)) ? geminiOffset.dy : 0;

    // מחזירים מערך מעודכן באותו מבנה בדיוק!
    return [
        x_center + dx,
        y_center + dy,
        width,
        height,
        confidence
    ];
});

const adjustedDetections = {
    ...detections,
    boxes: adjustedBoxes
};

// ציור התיבות המעודכנות
drawBoxes(adjustedDetections);

    }
    // 5. בקשה לפריים הבא - תמיד בסוף!
    requestAnimationFrame(detectFrame);
}

function drawBoxes(detections) {
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { boxes, scores } = detections;

    for (let i = 0; i < boxes.length; i++) {
        // המערך שראינו בלוגים שלך!
        const box = boxes[i];
        
        // הגנה: וודא שהאיבר הוא אכן מערך ויש בו לפחות 4 ערכים
        if (!Array.isArray(box) || box.length < 4) continue;

        // שליפת הנתונים לפי הסדר של YOLO
        let [x_center, y_center, width, height, box_score] = box;

        // שימוש ב-score הגבוה מבין השניים (זה שמהטנזור הנפרד או זה שבתוך הבוקס)
        const finalScore = box_score || scores[i];

        if (finalScore > CONF_THRESHOLD) {
            
            // 1. הגבלת המרכז לתוך גבולות הפריים (0 עד 1)
            const safeX = Math.max(0, Math.min(x_center, 1));
            const safeY = Math.max(0, Math.min(y_center, 1));

            // 2. חישוב מימדים סופיים בפיקסלים לפי גודל הקנבס האמיתי
            const w = width * canvas.width;
            const h = height * canvas.height;

            // 3. חישוב הפינה השמאלית העליונה (x1, y1) בפיקסלים
            const x1 = (safeX * canvas.width) - (w / 2);
            const y1 = (safeY * canvas.height) - (h / 2);

            // 4. ציור התיבה
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, w, h);

            // 5. הוספת אחוז הביטחון מעל הריבוע
            ctx.fillStyle = "#00FF00";
            ctx.font = "bold 16px Arial";
            ctx.fillText(`${(finalScore * 100).toFixed(0)}%`, x1, y1 > 20 ? y1 - 5 : 20);
        }
    }
}


setupApp();
