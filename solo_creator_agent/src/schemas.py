from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


Platform = Literal["xiaohongshu", "bilibili", "douyin", "wechat", "zhihu", "youtube", "tiktok", "instagram", "substack", "x"]
ContentType = Literal["tutorial", "story", "review", "listicle", "opinion", "case_study"]
TitleStyle = Literal["pain_point", "tutorial", "number", "story", "contrast", "result_oriented", "question"]
RevenueType = Literal["brand_ads", "platform_share", "course", "consulting", "membership", "affiliate", "reward", "service"]
RevenueStatus = Literal["received", "pending"]
CampaignStatus = Literal["negotiating", "confirmed", "scripting", "reviewing", "published", "reporting", "completed"]
PaymentStatus = Literal["unpaid", "deposit_received", "fully_paid", "overdue"]
InvoiceStatus = Literal["not_needed", "pending", "issued"]
ReportStatus = Literal["not_started", "generated", "sent"]
OutcomeMetric = Literal["view_rate", "like_rate", "favorite_rate", "follow_rate", "conversion_rate", "revenue"]
ExperimentGroup = Literal["treatment", "control"]
ProductCategory = Literal["apparel", "accessory", "digital_product", "hardware_device", "home_product", "course", "service"]
FeedbackSource = Literal["beta_test", "review", "comment", "survey", "support_ticket", "private_message"]
FeedbackSentiment = Literal["positive", "neutral", "negative"]
IssueType = Literal["usability", "pricing", "feature_request", "quality", "delivery", "trust", "content_clarity", "emotional_value", "design", "performance"]
Severity = Literal["low", "medium", "high"]


class CleanModel(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def empty_none_to_blank(cls, value):
        if value is None:
            return ""
        return value


class ContentRecord(BaseModel):
    content_id: str
    title: str
    platform: Platform
    body: str = ""
    tags: str = ""
    topic: str
    content_type: ContentType
    publish_time: datetime
    language: str = ""
    title_style: TitleStyle
    cover_style: str = "default"
    duration_sec: int = Field(ge=0)
    production_hours: float = Field(ge=0)
    followers_before: int = Field(default=0, ge=0)
    impressions: int = Field(default=0, ge=0)
    views: int = Field(ge=0)
    likes: int = Field(ge=0)
    favorites: int = Field(ge=0)
    comments: int = Field(ge=0)
    shares: int = Field(ge=0)
    completion_rate: float = Field(default=0, ge=0)
    new_followers: int = Field(ge=0)
    consultations: int = Field(ge=0)
    conversions: int = Field(ge=0)
    revenue: float = Field(ge=0)
    cost: float = Field(default=0, ge=0)
    ad_spend: float = Field(default=0, ge=0)
    is_sponsored: bool = False
    series_id: Optional[str] = ""
    parent_content_id: Optional[str] = ""
    content_similarity_group: Optional[str] = ""
    knowledge_domain: Optional[str] = ""
    difficulty_level: Optional[Literal["beginner", "intermediate", "advanced"]] = ""
    novelty_score: Optional[float] = Field(default=0, ge=0, le=1)
    duplication_risk: Optional[float] = Field(default=0, ge=0, le=1)
    user_fatigue_risk: Optional[float] = Field(default=0, ge=0, le=1)


class RevenueRecord(CleanModel):
    revenue_id: str
    date: date
    amount: float = Field(ge=0)
    revenue_type: RevenueType
    platform: Platform
    content_id: Optional[str] = ""
    client_name: Optional[str] = ""
    status: RevenueStatus
    note: str = ""


class CampaignRecord(CleanModel):
    campaign_id: str
    brand_name: str
    campaign_name: str
    platform: Platform
    deliverables: str
    price: float = Field(ge=0)
    deadline: date
    status: CampaignStatus
    payment_status: PaymentStatus
    invoice_status: InvoiceStatus
    revision_count: int = Field(ge=0)
    report_status: ReportStatus
    related_content_id: Optional[str] = ""


class ABTestRecord(BaseModel):
    experiment_id: str
    date: date
    platform: Platform
    topic: str
    treatment_name: str
    treatment_value: str
    control_value: str
    outcome_metric: OutcomeMetric
    group: ExperimentGroup
    content_id: str
    outcome_value: float
    covariates_json: str = "{}"


class ProductRecord(CleanModel):
    product_id: str
    product_name: str
    category: ProductCategory
    series_id: Optional[str] = ""
    launch_date: date
    price: float = Field(ge=0)
    cost: float = Field(ge=0)
    material: Optional[str] = ""
    color: Optional[str] = ""
    style: Optional[str] = ""
    size: Optional[str] = ""
    weight: Optional[float] = 0
    feature_tags: str = ""
    target_user: str = ""
    platform: Platform
    views: int = Field(ge=0)
    clicks: int = Field(ge=0)
    consultations: int = Field(ge=0)
    conversions: int = Field(ge=0)
    revenue: float = Field(ge=0)
    refund_count: int = Field(ge=0)
    review_count: int = Field(ge=0)
    avg_rating: float = Field(ge=0, le=5)
    is_new_version: bool = False
    parent_product_id: Optional[str] = ""


class FeedbackRecord(CleanModel):
    feedback_id: str
    user_id: str
    source_type: FeedbackSource
    related_content_id: Optional[str] = ""
    related_product_id: Optional[str] = ""
    user_segment: Optional[str] = ""
    country: Optional[str] = ""
    platform: Optional[Platform] = "xiaohongshu"
    feedback_text: str
    rating: Optional[float] = 0
    sentiment: FeedbackSentiment
    issue_type: Optional[IssueType] = "usability"
    severity: Severity
    created_at: datetime
    converted_after_feedback: Optional[bool] = False
    retained_after_feedback: Optional[bool] = False


class BetaTestRecord(CleanModel):
    beta_test_id: str
    product_id: Optional[str] = ""
    feature_name: Optional[str] = ""
    test_group: ExperimentGroup
    user_id: str
    user_segment: Optional[str] = ""
    invited_at: datetime
    experienced_at: Optional[datetime] = None
    feedback_submitted: bool = False
    rating: Optional[float] = 0
    activated: Optional[bool] = False
    retained_7d: Optional[bool] = False
    converted: Optional[bool] = False
    revenue: Optional[float] = 0
    notes: Optional[str] = ""
