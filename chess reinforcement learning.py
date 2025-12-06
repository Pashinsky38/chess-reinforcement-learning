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

# ---------- Configuration ----------
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
SIDEBAR_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_SIZE

WHITE = (238, 238, 210)
BLACK = (118, 150, 86)
HIGHLIGHT = (186, 202, 43)
TEXT_COLOR = (0, 0, 0)
BG_COLOR = (49, 46, 43)

# ---------- Small utilities ----------
_piece_map = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}


def material_score(board: chess.Board) -> int:
    """Simple material evaluation from White's perspective."""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            v = values[p.piece_type]
            score += v if p.color == chess.WHITE else -v
    return score


# ---------- Model ----------
class ChessNet(nn.Module):
    """Simple convolutional value network. Returns a scalar V(s) from White's perspective."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, 12, 8, 8)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1)  # (B,)


# ---------- Agent ----------
class ChessRLAgent:
    def __init__(self, learning_rate=1e-4, gamma=0.99, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = ChessNet().to(self.device)
        self.target_model = ChessNet().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma

        # Epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Replay buffer stores (state, reward, next_state, done)
        self.memory = deque(maxlen=20000)
        self.batch_size = 64

    def board_to_tensor(self, board: chess.Board) -> torch.FloatTensor:
        """Convert board to a (1,12,8,8) tensor. Uses chess.square_rank/file for correct orientation.
        Tensor ordering: channel x rank x file where rank, file in 0..7.
        Channels 0-5 white P,N,B,R,Q,K; 6-11 black p,n,b,r,q,k."""
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                r = chess.square_rank(sq)  # 0..7 (rank 1 -> 0)
                f = chess.square_file(sq)  # 0..7 (file a -> 0)
                ch = _piece_map[p.piece_type]
                if not p.color:
                    ch += 6
                tensor[ch, r, f] = 1.0
        t = torch.from_numpy(tensor).unsqueeze(0).to(self.device)  # (1,12,8,8)
        return t

    def get_legal_moves(self, board: chess.Board):
        return list(board.legal_moves)

    def evaluate_state(self, state_tensor: torch.FloatTensor) -> float:
        """Return scalar value (White's perspective) for a single state tensor."""
        self.model.eval()
        with torch.no_grad():
            v = self.model(state_tensor).item()
        self.model.train()
        return v

    def evaluate_move(self, board: chess.Board, move: chess.Move) -> float:
        b2 = board.copy()
        b2.push(move)
        s = self.board_to_tensor(b2)
        return self.evaluate_state(s)

    def select_move(self, board: chess.Board, training=True):
        legal = self.get_legal_moves(board)
        if not legal:
            return None

        # Exploration
        if training and random.random() < self.epsilon:
            return random.choice(legal)

        # Greedy: choose move maximizing the value for the side to move.
        # Model returns value from WHITE's perspective.
        move_vals = []
        for m in legal:
            v = self.evaluate_move(board, m)
            move_vals.append(v)

        if board.turn == chess.WHITE:
            idx = int(np.argmax(move_vals))
        else:
            idx = int(np.argmin(move_vals))
        return legal[idx]

    def remember(self, state, reward, next_state, done):
        # states are tensors already on device
        self.memory.append((state, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.cat([b[0] for b in batch], dim=0)  # (B,12,8,8)
        rewards = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.cat([b[2] for b in batch], dim=0)
        dones = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)

        # Compute targets: r + gamma * V(next) * (1-done)
        with torch.no_grad():
            next_vals = self.target_model(next_states)  # (B,)
        targets = rewards + self.gamma * next_vals * (1.0 - dones)

        current_vals = self.model(states)  # (B,)
        loss = nn.MSELoss()(current_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")


# ---------- GUI / Trainer (visual) ----------
class ChessGUI:
    def __init__(self, agent: ChessRLAgent, engine_path=None):
        pygame.init()
        self.agent = agent
        self.board = chess.Board()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess RL Agent Training")

        self.font = pygame.font.Font(None, 28)
        self.title_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 22)

        self.engine = None
        if engine_path and os.path.exists(engine_path):
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            except Exception as e:
                print("Engine start failed:", e)
                self.engine = None

        self.last_move = None
        self.game_count = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.current_skill_level = 1
        self.states = []  # list of (state_tensor, board_before_move)
        self.thinking = False
        self.load_pieces()

    def load_pieces(self):
        self.pieces = {}
        piece_names = ['P','N','B','R','Q','K','p','n','b','r','q','k']
        for pn in piece_names:
            self.pieces[pn] = self.create_piece_surface(pn)

    def create_piece_surface(self, piece_name):
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        is_white = piece_name.isupper()
        color = (255,255,255) if is_white else (0,0,0)
        outline = (0,0,0) if is_white else (255,255,255)
        center = (SQUARE_SIZE//2, SQUARE_SIZE//2)
        pygame.draw.circle(surface, color, center, 25)
        pygame.draw.circle(surface, outline, center, 25, 3)
        font = pygame.font.Font(None, 40)
        text = font.render(piece_name.upper(), True, outline)
        surface.blit(text, text.get_rect(center=center))
        return surface

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                if self.last_move:
                    frow = 7 - (self.last_move.from_square // 8)
                    fcol = self.last_move.from_square % 8
                    trow = 7 - (self.last_move.to_square // 8)
                    tcol = self.last_move.to_square % 8
                    if (row, col) == (frow, fcol) or (row, col) == (trow, tcol):
                        color = HIGHLIGHT
                pygame.draw.rect(self.screen, color, (col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        coord_font = pygame.font.Font(None, 20)
        for i in range(8):
            file_text = coord_font.render(chr(97 + i), True, TEXT_COLOR)
            self.screen.blit(file_text, (i*SQUARE_SIZE + SQUARE_SIZE - 15, BOARD_SIZE - 15))
            rank_text = coord_font.render(str(8 - i), True, TEXT_COLOR)
            self.screen.blit(rank_text, (5, i*SQUARE_SIZE + 5))

    def draw_pieces(self):
        for sq in chess.SQUARES:
            p = self.board.piece_at(sq)
            if p:
                row = 7 - (sq // 8)
                col = sq % 8
                s = self.pieces.get(p.symbol())
                if s:
                    self.screen.blit(s, (col*SQUARE_SIZE, row*SQUARE_SIZE))

    def draw_sidebar(self):
        sidebar_x = BOARD_SIZE
        pygame.draw.rect(self.screen, BG_COLOR, (sidebar_x, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        y = 20
        title = self.title_font.render("Chess RL Agent", True, (255,255,255))
        self.screen.blit(title, (sidebar_x + 20, y)); y += 60
        stats = [
            f"Game: {self.game_count}",
            f"Wins: {self.wins}",
            f"Losses: {self.losses}",
            f"Draws: {self.draws}",
            "",
            f"Epsilon: {self.agent.epsilon:.3f}",
            f"Bot Level: {self.current_skill_level}",
            "",
            f"To Move: {'White (AI)' if self.board.turn==chess.WHITE else 'Black (Bot)'}"
        ]
        for s in stats:
            t = self.font.render(s, True, (255,255,255))
            self.screen.blit(t, (sidebar_x + 20, y)); y += 35
        if self.last_move:
            y += 10
            lm = self.font.render('Last Move:', True, (255,255,255))
            self.screen.blit(lm, (sidebar_x + 20, y)); y += 30
            move_display = self.font.render(self.last_move.uci(), True, (100,255,100))
            self.screen.blit(move_display, (sidebar_x + 20, y))
        if self.thinking:
            y = WINDOW_HEIGHT - 50
            thinking_text = self.small_font.render('Thinking...', True, (255,255,100))
            self.screen.blit(thinking_text, (sidebar_x + 20, y))

    def play_game(self, delay=200):
        self.board = chess.Board()
        self.last_move = None
        self.states = []
        clock = pygame.time.Clock()

        while not self.board.is_game_over():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

            self.screen.fill(BG_COLOR)
            self.draw_board()
            self.draw_pieces()
            self.draw_sidebar()
            pygame.display.flip()

            if self.board.turn == chess.WHITE:
                self.thinking = True
                self.draw_sidebar(); pygame.display.flip()

                state = self.agent.board_to_tensor(self.board)
                move = self.agent.select_move(self.board, training=True)
                if move is None:
                    break
                # store state before move
                self.states.append((state, self.board.copy()))
                self.board.push(move)
                self.last_move = move
                self.thinking = False

            else:
                self.thinking = True
                self.draw_sidebar(); pygame.display.flip()
                if self.engine:
                    self.engine.configure({'Skill Level': self.current_skill_level})
                    res = self.engine.play(self.board, chess.engine.Limit(time=0.05))
                    self.board.push(res.move)
                    self.last_move = res.move
                else:
                    legal = list(self.board.legal_moves)
                    if legal:
                        m = random.choice(legal)
                        self.board.push(m)
                        self.last_move = m
                self.thinking = False

            pygame.time.delay(delay)
            clock.tick(60)

        # Evaluate final result
        res = self.board.result()
        if res == '1-0':
            reward = 1.0
            self.wins += 1
        elif res == '0-1':
            reward = -1.0
            self.losses += 1
        else:
            reward = 0.0
            self.draws += 1

        # Store transitions with a basic shaped reward: change in material + final outcome
        for i, (state, board_before) in enumerate(self.states):
            # board_before is BEFORE the AI moved; compute board after that move
            # The board AFTER AI moved is states[i+1] board_before OR final board
            if i < len(self.states) - 1:
                board_after = self.states[i+1][1]
            else:
                board_after = self.board
            material_before = material_score(board_before)
            material_after = material_score(board_after)
            shaped = (material_after - material_before) / 10.0  # small shaping
            done = (i == len(self.states) - 1)
            # final reward added to last transition
            r = shaped + (reward if done else 0.0)
            s = state  # tensor on device
            ns = self.agent.board_to_tensor(board_after)
            self.agent.remember(s, r, ns, float(done))

        return res

    def train_with_visualization(self, episodes=100, update_target_every=10, save_every=50):
        print(f"Training on {self.agent.device}")
        print(f"Engine: {'Available' if self.engine else 'Not available'}")

        for ep in range(episodes):
            self.game_count = ep + 1
            self.current_skill_level = min(1 + ep // 10, 20)
            result = self.play_game(delay=200)
            if result is None:
                break
            # train
            self.agent.replay()
            if ep % update_target_every == 0:
                self.agent.update_target_model()
            if ep % save_every == 0 and ep > 0:
                self.agent.save_model(f"chess_model_ep{ep}.pt")

            # brief game over display
            self.screen.fill(BG_COLOR)
            self.draw_board(); self.draw_pieces(); self.draw_sidebar()
            font = pygame.font.Font(None, 48)
            if result == '1-0':
                msg = font.render('AI WINS!', True, (0,255,0))
            elif result == '0-1':
                msg = font.render('AI LOSES', True, (255,0,0))
            else:
                msg = font.render('DRAW', True, (255,255,0))
            self.screen.blit(msg, msg.get_rect(center=(BOARD_SIZE//2, BOARD_SIZE//2)))
            pygame.display.flip()
            pygame.time.delay(800)

        if self.engine:
            self.engine.quit()
        pygame.quit()


# ---------- Example usage ----------
if __name__ == '__main__':
    agent = ChessRLAgent(learning_rate=1e-4, gamma=0.95)
    # agent.load_model('chess_model_final.pt')
    stockfish_path = None  # e.g., 'C:/stockfish/stockfish.exe'
    gui = ChessGUI(agent, engine_path=stockfish_path)
    gui.train_with_visualization(episodes=100)
    agent.save_model('chess_model_final.pt')
