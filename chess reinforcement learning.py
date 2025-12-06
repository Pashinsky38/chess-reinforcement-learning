import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import pygame
import sys

# Initialize Pygame
pygame.init()

# Colors
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)
HIGHLIGHT = (186, 202, 43)
TEXT_COLOR = (0, 0, 0)
BG_COLOR = (49, 46, 43)

# Board settings
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
SIDEBAR_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_SIZE

class ChessNet(nn.Module):
    """Neural network to evaluate chess positions"""
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class ChessRLAgent:
    def __init__(self, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.target_model = ChessNet().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
    def board_to_tensor(self, board):
        """Convert chess board to neural network input tensor"""
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                piece_type = piece_map[piece.piece_type]
                channel = piece_type if piece.color == chess.WHITE else piece_type + 6
                tensor[channel, rank, file] = 1.0
        
        return torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
    
    def get_legal_moves(self, board):
        """Get list of legal moves"""
        return list(board.legal_moves)
    
    def evaluate_move(self, board, move):
        """Evaluate a specific move"""
        board_copy = board.copy()
        board_copy.push(move)
        state = self.board_to_tensor(board_copy)
        
        with torch.no_grad():
            value = self.model(state).item()
        
        return value
    
    def select_move(self, board, training=True):
        """Select a move using epsilon-greedy policy"""
        legal_moves = self.get_legal_moves(board)
        
        if not legal_moves:
            return None
        
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        move_values = []
        for move in legal_moves:
            value = self.evaluate_move(board, move)
            move_values.append(-value if board.turn == chess.WHITE else value)
        
        best_idx = np.argmax(move_values)
        return legal_moves[best_idx]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            
            if not done:
                next_value = self.target_model(next_state).item()
                target = reward + self.gamma * next_value
            
            current_value = self.model(state)
            loss = nn.MSELoss()(current_value, torch.FloatTensor([[target]]).to(self.device))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Update target network"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")

class ChessGUI:
    def __init__(self, agent, engine_path=None):
        self.agent = agent
        self.board = chess.Board()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess RL Agent Training")
        
        self.font = pygame.font.Font(None, 28)
        self.title_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 22)
        
        self.engine = None
        if engine_path and os.path.exists(engine_path):
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        
        self.last_move = None
        self.game_count = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.current_skill_level = 1
        self.states = []
        self.thinking = False
        
        # Load piece images
        self.load_pieces()
    
    def load_pieces(self):
        """Load chess piece images"""
        self.pieces = {}
        piece_names = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        
        # Create simple colored circles for pieces (you can replace with actual images)
        for piece_name in piece_names:
            self.pieces[piece_name] = self.create_piece_surface(piece_name)
    
    def create_piece_surface(self, piece_name):
        """Create a simple piece representation"""
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        
        is_white = piece_name.isupper()
        color = (255, 255, 255) if is_white else (0, 0, 0)
        outline_color = (0, 0, 0) if is_white else (255, 255, 255)
        
        # Draw circle
        center = (SQUARE_SIZE // 2, SQUARE_SIZE // 2)
        pygame.draw.circle(surface, color, center, 25)
        pygame.draw.circle(surface, outline_color, center, 25, 3)
        
        # Draw piece letter
        font = pygame.font.Font(None, 40)
        text = font.render(piece_name.upper(), True, outline_color)
        text_rect = text.get_rect(center=center)
        surface.blit(text, text_rect)
        
        return surface
    
    def draw_board(self):
        """Draw the chess board"""
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                
                # Highlight last move
                if self.last_move:
                    from_square = self.last_move.from_square
                    to_square = self.last_move.to_square
                    from_row, from_col = 7 - (from_square // 8), from_square % 8
                    to_row, to_col = 7 - (to_square // 8), to_square % 8
                    
                    if (row, col) == (from_row, from_col) or (row, col) == (to_row, to_col):
                        color = HIGHLIGHT
                
                pygame.draw.rect(self.screen, color, 
                               (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        
        # Draw coordinates
        coord_font = pygame.font.Font(None, 20)
        for i in range(8):
            # Files (a-h)
            file_text = coord_font.render(chr(97 + i), True, TEXT_COLOR)
            self.screen.blit(file_text, (i * SQUARE_SIZE + SQUARE_SIZE - 15, BOARD_SIZE - 15))
            
            # Ranks (1-8)
            rank_text = coord_font.render(str(8 - i), True, TEXT_COLOR)
            self.screen.blit(rank_text, (5, i * SQUARE_SIZE + 5))
    
    def draw_pieces(self):
        """Draw all pieces on the board"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                piece_symbol = piece.symbol()
                
                piece_surface = self.pieces.get(piece_symbol)
                if piece_surface:
                    self.screen.blit(piece_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))
    
    def draw_sidebar(self):
        """Draw the information sidebar"""
        sidebar_x = BOARD_SIZE
        pygame.draw.rect(self.screen, BG_COLOR, (sidebar_x, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        
        y_offset = 20
        
        # Title
        title = self.title_font.render("Chess RL Agent", True, (255, 255, 255))
        self.screen.blit(title, (sidebar_x + 20, y_offset))
        y_offset += 60
        
        # Stats
        stats = [
            f"Game: {self.game_count}",
            f"Wins: {self.wins}",
            f"Losses: {self.losses}",
            f"Draws: {self.draws}",
            "",
            f"Epsilon: {self.agent.epsilon:.3f}",
            f"Bot Level: {self.current_skill_level}",
            "",
            f"To Move: {'White (AI)' if self.board.turn == chess.WHITE else 'Black (Bot)'}",
        ]
        
        for stat in stats:
            text = self.font.render(stat, True, (255, 255, 255))
            self.screen.blit(text, (sidebar_x + 20, y_offset))
            y_offset += 35
        
        # Last move
        if self.last_move:
            y_offset += 10
            move_text = self.font.render("Last Move:", True, (255, 255, 255))
            self.screen.blit(move_text, (sidebar_x + 20, y_offset))
            y_offset += 30
            
            move_str = self.last_move.uci()
            move_display = self.font.render(move_str, True, (100, 255, 100))
            self.screen.blit(move_display, (sidebar_x + 20, y_offset))
        
        # Thinking indicator
        if self.thinking:
            y_offset = WINDOW_HEIGHT - 50
            thinking_text = self.small_font.render("Thinking...", True, (255, 255, 100))
            self.screen.blit(thinking_text, (sidebar_x + 20, y_offset))
    
    def play_game(self, delay=500):
        """Play one game with visual display"""
        self.board = chess.Board()
        self.last_move = None
        self.states = []
        clock = pygame.time.Clock()
        
        while not self.board.is_game_over():
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
            
            # Draw everything
            self.screen.fill(BG_COLOR)
            self.draw_board()
            self.draw_pieces()
            self.draw_sidebar()
            pygame.display.flip()
            
            # Make move
            if self.board.turn == chess.WHITE:
                # AI's turn
                self.thinking = True
                self.draw_sidebar()
                pygame.display.flip()
                
                state = self.agent.board_to_tensor(self.board)
                move = self.agent.select_move(self.board, training=True)
                
                if move is None:
                    break
                
                self.board.push(move)
                self.last_move = move
                self.states.append((state, move, self.board.copy()))
                self.thinking = False
                
            else:
                # Engine's turn
                self.thinking = True
                self.draw_sidebar()
                pygame.display.flip()
                
                if self.engine:
                    self.engine.configure({"Skill Level": self.current_skill_level})
                    result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
                    self.board.push(result.move)
                    self.last_move = result.move
                else:
                    legal_moves = list(self.board.legal_moves)
                    if legal_moves:
                        move = random.choice(legal_moves)
                        self.board.push(move)
                        self.last_move = move
                
                self.thinking = False
            
            # Delay between moves
            pygame.time.delay(delay)
            clock.tick(60)
        
        # Game over - determine result
        result = self.board.result()
        if result == "1-0":
            reward = 1.0
            self.wins += 1
        elif result == "0-1":
            reward = -1.0
            self.losses += 1
        else:
            reward = 0.0
            self.draws += 1
        
        # Store experiences
        for i, (state, move, board_state) in enumerate(self.states):
            next_state = self.agent.board_to_tensor(board_state)
            done = (i == len(self.states) - 1)
            self.agent.remember(state, move, reward, next_state, done)
        
        return result
    
    def train_with_visualization(self, episodes=100, update_target_every=10, save_every=50):
        """Train the agent with live visualization"""
        print(f"Training on {self.agent.device}")
        print(f"Engine: {'Available' if self.engine else 'Not available (using random moves)'}")
        
        for episode in range(episodes):
            self.game_count = episode + 1
            self.current_skill_level = min(1 + episode // 10, 20)
            
            result = self.play_game(delay=300)  # 300ms delay between moves
            
            if result is None:  # User closed window
                break
            
            # Train
            self.agent.replay()
            
            # Update target network
            if episode % update_target_every == 0:
                self.agent.update_target_model()
            
            # Save model
            if episode % save_every == 0 and episode > 0:
                self.agent.save_model(f"chess_model_ep{episode}.pt")
            
            # Show game over screen briefly
            self.screen.fill(BG_COLOR)
            self.draw_board()
            self.draw_pieces()
            self.draw_sidebar()
            
            # Game over text
            game_over_font = pygame.font.Font(None, 48)
            if result == "1-0":
                text = game_over_font.render("AI WINS!", True, (0, 255, 0))
            elif result == "0-1":
                text = game_over_font.render("AI LOSES", True, (255, 0, 0))
            else:
                text = game_over_font.render("DRAW", True, (255, 255, 0))
            
            text_rect = text.get_rect(center=(BOARD_SIZE // 2, BOARD_SIZE // 2))
            self.screen.blit(text, text_rect)
            pygame.display.flip()
            pygame.time.delay(1500)
        
        if self.engine:
            self.engine.quit()
        
        pygame.quit()

# Example usage
if __name__ == "__main__":
    # Create agent
    agent = ChessRLAgent(learning_rate=0.0001, gamma=0.95)
    
    # Optional: Load existing model
    # agent.load_model("chess_model_ep1000.pt")
    
    # Create GUI trainer
    # Replace with your Stockfish path, or leave None for random opponent
    stockfish_path = None  # e.g., "/usr/local/bin/stockfish" or "C:/stockfish/stockfish.exe"
    
    gui = ChessGUI(agent, engine_path=stockfish_path)
    
    # Train with visualization
    gui.train_with_visualization(episodes=100)
    
    # Save final model
    agent.save_model("chess_model_final.pt")