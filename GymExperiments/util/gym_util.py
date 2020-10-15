def create_video_callable(save_ith_episode):
    def video_callable(episode):
        return True if ((episode+1) % save_ith_episode == 0) else False
    return video_callable 