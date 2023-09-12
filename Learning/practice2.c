#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <conio.h>

#define N 4
#define TARGET 2048

void init(int board[N][N]);
int move(int board[N][N], int direction);
void addRandomTile(int board[N][N]);
void display(int board[N][N]);
int hasWon(int board[N][N]);
int hasMoves(int board[N][N]);

int main()
{
    int board[N][N], direction;

    srand(time(NULL));
    init(board);

    while (!hasWon(board) && hasMoves(board))
    {
        display(board);
        printf("Enter direction (w:0, a:1, s:2, d:3): ");
        scanf(" %c", &direction);

        switch (direction)
        {
        case 'w':
            direction = 0;
            break;
        case 'a':
            direction = 1;
            break;
        case 's':
            direction = 2;
            break;
        case 'd':
            direction = 3;
            break;
        default:
            printf("Invalid direction! Try again.\n");
            continue;
        }

        if (move(board, direction))
        {
            addRandomTile(board);
        }
        else
        {
            printf("Invalid move! Try again.\n");
        }
    }

    display(board);

    if (hasWon(board))
    {
        printf("You won!\n");
    }
    else
    {
        printf("You lost!\n");
    }

    return 0;
}

void init(int board[N][N])
{
    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            board[i][j] = 0;
        }
    }

    addRandomTile(board);
    addRandomTile(board);
}

int move(int board[N][N], int direction)
{
    int i, j, k, moved = 0;

    for (i = 0; i < N; i++)
    {
        for (j = (direction % 2 == 0 ? N - 1 : 0); j >= 0 && j < N; j += (direction % 2 == 0 ? -1 : 1))
        {
            if (board[direction < 2 ? i : j][direction < 2 ? j : i] != 0)
            {
                for (k = j + (direction % 2 == 0 ? 1 : -1); k >= 0 && k < N; k += (direction % 2 == 0 ? 1 : -1))
                {
                    if (board[direction < 2 ? i : k][direction < 2 ? k : i] == 0)
                    {
                        continue;
                    }
                    else if (board[direction < 2 ? i : k][direction < 2 ? k : i] == board[direction < 2 ? i : j][direction < 2 ? j : i])
                    {
                        board[direction < 2 ? i : k][direction < 2 ? k : i] *= 2;
                        board[direction < 2 ? i : j][direction < 2 ? j : i] = 0;
                        moved = 1;
                    }
                    break;
                }
            }
        }
    }

    return moved;
}

void addRandomTile(int board[N][N])
{
    int i, j, k, emptyTiles = 0, randTile;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (board[i][j] == 0)
            {
                emptyTiles++;
            }
        }
    }

    if (emptyTiles == 0)
    {
        return;
    }

    randTile = rand() % emptyTiles;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (board[i][j] == 0)
            {
                if (randTile == 0)
                {
                    board[i][j] = (rand() % 2 + 1) * 2;
                    return;
                }
                randTile--;
            }
        }
    }
}

void display(int board[N][N])
{
    int i, j;

    system("cls");

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%4d ", board[i][j]);
        }
        printf("\n");
    }
}

int hasWon(int board[N][N])
{
    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (board[i][j] == TARGET)
            {
                return 1;
            }
        }
    }

    return 0;
}

int hasMoves(int board[N][N])
{
    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (board[i][j] == 0)
            {
                return 1;
            }
            if (i < N - 1 && board[i][j] == board[i + 1][j])
            {
                return 1;
            }
            if (j < N - 1 && board[i][j] == board[i][j + 1])
            {
                return 1;
            }
        }
    }

    return 0;
}



