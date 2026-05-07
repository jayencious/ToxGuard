"use client";

import React, { useState } from "react";

interface Comment {
  id: number;
  text: string;
  action: "allow" | "hide" | "delete";
  confidence: number;
};

export default function Home() {
  const [inputText, setInputText] = useState("");
  const [comments, setComments] = useState<Comment[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!inputText.trim()) return;

    setIsSubmitting(true);

    try {
      // Calling our FastAPI Engine
      const res = await fetch(
        "http://127.0.0.1:8000/moderate-text",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(
            { text: inputText }
          ),
        }
      );

      const data = await res.json();

      // Creating the new comment object based on the API response
      const newComment: Comment = {
        id: Date.now(),
        text: data.original_text,
        action: data.recommended_action,
        confidence: data.confidence_score,
      };

      // Add to UI
      setComments([newComment, ...comments]);
      setInputText("");
    } catch (err) {
      console.error("Failed to reach API:", err);
      alert("Error connecting to the Content Moderation API.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      className="min-h-screen bg-gray-50 py-10 px-4 flex-justify-center"
    >
      <div
        className="max-w-2xl w-full bg-white rounded-xl shadow-lg p-6"
      >

        {/* Mock/Dummy Blog Post */}
        <h1
          className="text-3xl font-bold text-gray-800 mb-4"
        >
          How to Build a REST API
        </h1>
        <p
          className="text-gray-600 mb-8 border-b pb-8"
        >
          Welcome to my blog! Today, we are going to discuss about Software Engineering and API (Application Programming Interface) design.
          Let me know your thoughts in the comments below.
        </p>

        {/* Comment Input Form */}
        <form
          onSubmit={handleSubmit}
          className="mb-8"
        >
          <textarea
            className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 outline-none resize-none text-black"
            rows={3}
            placeholder="Write a comment..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
          <button
            type="submit"
            disabled={isSubmitting}
            className="mt-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors disabled:opacity-50"
          >
            {isSubmitting
              ? "Analyzing..."
              : "Post Comment"
            }
          </button>
        </form>

        {/* Comment Feed */}
        <div
          className="space-y-4"
        >
          <h3
            className="text-xl font-semibold text-gray-800"
          >
            Comments ({comments.length})
          </h3>

          {comments.length === 0 && (
            <p
              className="text-gray-500 italic"
            >
              No comments yet. Be the first!
            </p>
          )}

          {comments.map(
            (comment) => (
              <div
                key={comment.id}
                className="p-4 rounded-lg border"
              >
                {/* Dynamic rendering based on the ML model's decision */}
                {comment.action === "allow" && (
                  <div>
                    <p
                      className="text-gray-800"
                    >
                      {comment.text}
                    </p>
                    <span
                      className="text-xs text-green-600 font-semibold mt-2 block"
                    >
                      ✅ Comment Approved! (Toxicity: {(comment.confidence * 100).toFixed(2)}%)
                    </span>
                  </div>
                )}

                {comment.action === "hide" && (
                  <div>
                    <p
                      className="text-gray-500 italic line-through"
                    >
                      This comment has been hidden. Pending manual review.
                    </p>
                    <span
                      className="text-xs text-yellow-600 font-semibold mt-2 block"
                    >
                      ⚠️ Comment Flagged! (Toxicity: {(comment.confidence * 100).toFixed(2)}%)
                    </span>
                  </div>
                )}

                {comment.action === "delete" && (
                  <div>
                    <p
                      className="text-red-500 font-medium flex items-center gap-2"
                    >
                      🚫 [Comment automatically removed through Moderation!]
                    </p>
                    <span
                      className="text-xs text-red-600 font-semibold mt-2 block"
                    >
                      Action: Comment Deleted! (Toxicity: {(comment.confidence * 100).toFixed(2)}%)
                    </span>
                  </div>
                )}
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
};