# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from ai_script_summarizer_improved import ImprovedScriptSummarizerAI
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Script Summarizer API",
    description="API cho việc tóm tắt kịch bản phim thành các tập A4",
    version="1.0.0"
)

# Cấu hình CORS
origins = [
    "http://localhost:2312",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Cho phép những origin này
    allow_credentials=True,
    allow_methods=["*"],              # Cho phép tất cả các method: GET, POST, PUT, DELETE,...
    allow_headers=["*"],              # Cho phép tất cả headers
)

# Initialize AI
ai_summarizer = ImprovedScriptSummarizerAI(
    mongodb_uri="mongodb+srv://nguyenvanninh:MM1onTUoA1sF4AnZ@onebizai.tt3rs.mongodb.net/",
    db_name="test"
)

# Pydantic models
class ScriptSummaryRequest(BaseModel):
    script: str
    episodeNumber: int = 3
    scriptId: Optional[str] = None  # ID của script trong DB
    
    class Config:
        schema_extra = {
            "example": {
                "script": "Đây là một câu chuyện tình yêu đẹp. Anh và em gặp nhau trong một ngày mưa...",
                "episodeNumber": 3,
                "scriptId": "60f7b3b3b3b3b3b3b3b3b3b3"
            }
        }

class FeedbackTrainingRequest(BaseModel):
    feedbacks: Optional[List[str]] = None  # Nếu None sẽ lấy từ DB
    useDbFeedbacks: bool = True  # Có sử dụng feedback từ DB không
    
    class Config:
        schema_extra = {
            "example": {
                "feedbacks": [
                    "Tóm tắt rất hay và chi tiết",
                    "Thiếu một số cảnh quan trọng",
                    "Chất lượng tốt, cảm ơn"
                ],
                "useDbFeedbacks": True
            }
        }

class SummaryFeedbackRequest(BaseModel):
    aiSummaryId: str
    rating: float  # 1-5
    feedbackText: Optional[str] = ""
    
    class Config:
        schema_extra = {
            "example": {
                "aiSummaryId": "60f7b3b3b3b3b3b3b3b3b3b3",
                "rating": 4.5,
                "feedbackText": "Tóm tắt rất chi tiết và chính xác"
            }
        }

class ScriptSummaryResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class TrainingResponse(BaseModel):
    success: bool
    feedback_processed: Optional[int] = None
    improvements: Optional[dict] = None
    training_id: Optional[str] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Script Summarizer API is running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now()
    }

@app.post("/api/summarize", response_model=ScriptSummaryResponse)
async def create_summary(request: ScriptSummaryRequest):
    """
    Tạo bản tóm tắt kịch bản thành các tập A4
    
    - **script**: Nội dung kịch bản gốc
    - **episodeNumber**: Số tập muốn chia (mặc định: 3)
    - **scriptId**: ID của script trong database (optional)
    
    Trả về tóm tắt chi tiết cho từng tập với độ dài 1 trang A4 (550-650 từ)
    """
    try:
        start_time = datetime.now()
        
        # Validate input
        if not request.script or len(request.script.strip()) < 100:
            raise HTTPException(
                status_code=400, 
                detail="Kịch bản quá ngắn. Cần ít nhất 100 ký tự."
            )
        
        if request.episodeNumber < 1 or request.episodeNumber > 20:
            raise HTTPException(
                status_code=400,
                detail="Số tập phải từ 1 đến 20"
            )
        
        # Process summarization
        result = ai_summarizer.summarize_script(
            script=request.script,
            episode_number=request.episodeNumber,
            script_id=request.scriptId
        )
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        processing_time = (datetime.now() - start_time).total_seconds()

        return ScriptSummaryResponse(
            success=True,
            data=result,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@app.post("/api/train", response_model=TrainingResponse)
async def train_model(request: FeedbackTrainingRequest, background_tasks: BackgroundTasks):
    """
    Huấn luyện lại mô hình AI từ feedback người dùng
    
    - **feedbacks**: Danh sách feedback từ người dùng (optional)
    - **useDbFeedbacks**: Có sử dụng feedback từ database không
    
    Mô hình sẽ học từ feedback để cải thiện chất lượng tóm tắt
    """
    try:
        feedback_list = None
        
        if request.useDbFeedbacks:
            # Sử dụng feedback từ DB
            feedback_list = None  # Để AI tự lấy từ DB
        else:
            # Sử dụng feedback từ request
            if not request.feedbacks or len(request.feedbacks) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Cần ít nhất 1 feedback để huấn luyện"
                )
            
            # Validate feedback content
            feedback_list = [
                feedback.strip() for feedback in request.feedbacks 
                if feedback.strip() and len(feedback.strip()) >= 5
            ]
            
            if len(feedback_list) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Không có feedback hợp lệ (cần ít nhất 5 ký tự)"
                )
        
        # Process training
        training_result = ai_summarizer.train_from_feedback(feedback_list)
        
        if 'error' in training_result:
            raise HTTPException(status_code=500, detail=training_result['error'])
        
        return TrainingResponse(
            success=True,
            feedback_processed=training_result.get('feedback_processed'),
            improvements=training_result.get('improvements'),
            training_id=training_result.get('training_id')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi huấn luyện: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(request: SummaryFeedbackRequest):
    """
    Gửi feedback cho AI summary
    
    - **aiSummaryId**: ID của AI summary
    - **rating**: Đánh giá từ 1-5
    - **feedbackText**: Nội dung feedback (optional)
    """
    try:
        # Validate rating
        if not (1 <= request.rating <= 5):
            raise HTTPException(
                status_code=400,
                detail="Rating phải từ 1 đến 5"
            )
        
        # Update feedback
        ai_summarizer.update_summary_feedback(
            ai_summary_id=request.aiSummaryId,
            feedback_rating=request.rating,
            feedback_text=request.feedbackText
        )
        
        return {
            "success": True,
            "message": "Feedback đã được ghi nhận",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi gửi feedback: {str(e)}")

@app.get("/api/stats")
async def get_statistics():
    """
    Lấy thống kê về hoạt động của AI
    """
    try:
        stats = ai_summarizer.get_training_statistics()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy thống kê: {str(e)}")
    
@app.get("/api/summaries")
async def get_all_summaries(limit: int = 100):
    """
    Lấy toàn bộ danh sách AI summaries
    
    - **limit**: Số lượng bản ghi tối đa (mặc định: 100)
    """
    try:
        summaries = ai_summarizer.get_all_summaries(limit=limit)
        return {
            "success": True,
            "count": len(summaries),
            "data": summaries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn summaries: {str(e)}")

@app.get("/api/categories")
async def get_categories():
    """
    Lấy danh sách categories từ database
    """
    try:
        categories = ai_summarizer.get_categories_from_db()
        return {
            "success": True,
            "data": categories,
            "count": len(categories)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy categories: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    Kiểm tra sức khỏe của hệ thống
    """
    try:
        # Test database connection
        stats = ai_summarizer.get_training_statistics()
        
        return {
            "status": "healthy",
            "database": "connected",
            "ai_model": "ready",
            "timestamp": datetime.now(),
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now()
        }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7000,
        reload=True,
        log_level="info"
    )
