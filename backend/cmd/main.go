package main

import (
	"log"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"

	"github.com/LeiLiLab/InfiniSST/backend/internal/api"
	"github.com/LeiLiLab/InfiniSST/backend/internal/service"
)

func main() {
	r := gin.Default()

	// Setup CORS
	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	config.AllowHeaders = []string{"Origin", "Content-Length", "Content-Type", "Authorization"}
	r.Use(cors.New(config))

	// Create services
	translationService := service.NewTranslationService()

	// Create API handlers
	apiHandler := api.NewHandler(translationService)

	// Setup routes
	r.POST("/api/translate", apiHandler.TranslateAudio)
	r.GET("/api/translate/stream", apiHandler.StreamTranslation)
	r.POST("/api/youtube", apiHandler.TranslateYoutubeVideo)

	// Start server
	log.Println("Starting server on :8080")
	if err := r.Run(":8080"); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
} 