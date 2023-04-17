#include <mpi.h>
// #include <chrono>
#include <fstream>
#include <vector>
#include <sstream>
// #include <string>
// #include <algorithm>
#include <tuple>
#include <iostream>
#include <deque>
#include <random>
#include <unistd.h>
#include <climits>
#include <algorithm>
#include <functional>
#define GREEN true
#define RED false

#define TOKEN_NONE 0
#define TOKEN_GREEN 1
#define TOKEN_RED 2


int tag_terminate =1;
int tag_token =2;
int tag_request_work=3;
int tag_response_work_size=4;
int tag_response_work_data=5;
int tag_response_parents_size=6;
int tag_response_parents_data=7;
int tag_new_UB=8;
int tag_no_backtracking=9;

unsigned int SHARE_DEPTH_LIMIT;

typedef struct task_s {
        unsigned int id;
        unsigned int duration;
        unsigned int release;
        unsigned int deadline;
        unsigned int cost_lb;
        task_s(unsigned int id, unsigned int duration, unsigned int release, unsigned int deadline) :
            id(id), duration(duration), release(release), deadline(deadline),cost_lb(release+duration) {}
        task_s() : id(0), duration(0), release(0), deadline(0),cost_lb(0) {}
} task_t;


MPI_Datatype mpi_task_type;

void create_task_type()
{
    const int nitems = 5;
    int          blocklengths[nitems] = {1,1,1,1,1};
    MPI_Datatype types[nitems] = {MPI_UNSIGNED, MPI_UNSIGNED,MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED};
    MPI_Aint     offsets[nitems];

    offsets[0] = offsetof(task_t, id);
    offsets[1] = offsetof(task_t, duration);
    offsets[2] = offsetof(task_t, release);
    offsets[3] = offsetof(task_t, deadline);
    offsets[4] = offsetof(task_t, cost_lb);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_task_type);
    MPI_Type_commit(&mpi_task_type);
}

typedef struct task_node_s {
        unsigned int id;
        unsigned int cost;
        unsigned int depth;
        task_node_s(unsigned int id, unsigned int cost, unsigned int depth) : id(id), cost(cost), depth(depth) {}
        task_node_s() : id(0), cost(0), depth(0) {}
} task_node_t;

MPI_Datatype mpi_task_node_type;

void create_task_node_type()
{
    int          blocklengths[3] = {1,1,1};
    MPI_Datatype types[3] = {MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED};
    MPI_Aint     offsets[3];

    offsets[0] = offsetof(task_node_t, id);
    offsets[1] = offsetof(task_node_t, cost);
    offsets[2] = offsetof(task_node_t, depth);

    MPI_Type_create_struct(3, blocklengths, offsets, types, &mpi_task_node_type);
    MPI_Type_commit(&mpi_task_node_type);

}


void printHelpPage(char *program) {
    std::cout << "Solve scheduling with Bratley's algorithm." << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "\t" << program << " INPUT_PATH OUTPUT_PATH" << std::endl << std::endl;
}

void check_args(int argc, char **argv, int myRank)
{
    if (argc == 1) {
        if (myRank == 0) {
            printHelpPage(argv[0]);
        }
        MPI_Finalize();
        exit(0);
    } 
    if (argc != 3) {
        if (myRank == 0) {
            printHelpPage(argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }
}

std::vector<task_t> readInstance(const std::string& instanceFileName) {
    size_t n = 0;
    std::vector<task_t> tasks;

    std::ifstream file(instanceFileName);
    if (file.is_open()) {
        int lineId = 0;
        std::string line;
        while (std::getline(file, line) and lineId <= n ) {
            std::stringstream ss(line);
            if (lineId == 0) {
                ss >> n;
                tasks.reserve(n);
                lineId++;
            } else {
                int p,r,d;
                ss >> p >> r >> d;
                tasks.emplace_back(lineId-1,p,r,d);
                lineId++;
            }
        }
        file.close();
    } else {
        throw std::runtime_error("It is not possible to open instance file!\n");
    }
    return tasks;
}

unsigned int validate(const std::vector<task_t>& tasks)
{
    unsigned int max_deadline = 0;
    unsigned int sum = 0;
    for(const auto& t : tasks)
    {
        sum+=t.duration;
        if(t.deadline > max_deadline)
            max_deadline = t.deadline;

        if(t.cost_lb > t.deadline)
        {
            return 0;
        }
    }

    if(sum > max_deadline)
        return 0;

    return max_deadline;
}

void writeOutput(const std::string& outputFileName, const std::vector<unsigned int>& start_times) {
    std::ofstream file(outputFileName);
    if (file.is_open()) {
        if(start_times.empty()) {
            file << "-1";
        }
        else {
            for (auto start_time: start_times) {
                file << start_time << '\n';
            }
        }
    }
}



void distributeInitialData(std::vector<task_t> tasks, int UB)
{
    int N = tasks.size();
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&UB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(N !=0)
        MPI_Bcast(tasks.data(), N, mpi_task_type, 0, MPI_COMM_WORLD);
}


std::tuple<std::vector<task_t>, unsigned int> receiveInitialData()
{
    int N = 0;
    int UB = 0;

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&UB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<task_t> tasks;
    if(N !=0)
    {
        tasks.resize(N);
        MPI_Bcast(tasks.data(), N, mpi_task_type, 0, MPI_COMM_WORLD);
    }

    return std::make_tuple(tasks, UB);
}



class MessageChannel_Token
{
private:
    MPI_Request r_send;
    MPI_Request r_recv;
    MPI_Status  s_send;
    MPI_Status  s_recv;

    int data_snd;
    int data_rcv;
    int flag;

    int src;
    int dst;

    bool have_sent_before = false;

public:
    MessageChannel_Token(int src, int dst): src(src), dst(dst) {
        MPI_Irecv(&data_rcv, 1, MPI_INT, src, tag_token, MPI_COMM_WORLD, &r_recv);
    }

    void send(int _data) {
        if(have_sent_before)
            MPI_Wait(&r_send, MPI_STATUS_IGNORE); //wait for prev message to be sent .... is it really needed ?

        data_snd = _data;
        MPI_Isend(&data_snd, 1, MPI_INT, dst, tag_token, MPI_COMM_WORLD, &r_send);
        have_sent_before = true;
    }

    bool receive(int &data) {
        MPI_Test(&r_recv, &flag, &s_recv);

        if(flag == 1 )
        {
            data = data_rcv;
            MPI_Irecv(&data_rcv, 1, MPI_INT, src, tag_token, MPI_COMM_WORLD, &r_recv);
            return true;
        }

        return false;

    }

    ~MessageChannel_Token() = default;
};


class MessageChannel_Terminate
{
private:
    int worldSize;
    MPI_Request r_recv;

    bool data = true;

    int flag;

    int src;
    int tag;

public:
    MessageChannel_Terminate(int rank, int worldSize) : worldSize(worldSize){

        if(rank != 0)
            MPI_Irecv(&data, 1, MPI_C_BOOL, 0, tag_terminate, MPI_COMM_WORLD, &r_recv);
    }

    void send() {
        MPI_Request r_send[worldSize-1];
        for (size_t i = 1; i < worldSize; i++)
        {
            MPI_Isend(&data, 1, MPI_C_BOOL, i, tag_terminate, MPI_COMM_WORLD, r_send + i-1);
        }

        MPI_Waitall(worldSize-1, r_send, MPI_STATUSES_IGNORE);
    }

    bool receive() {
        MPI_Test(&r_recv, &flag, MPI_STATUS_IGNORE);

        if(flag == 1 )
        {
            return true;
        }

        return false;

    }

    ~MessageChannel_Terminate() = default;
};

class MessageChannel_OneToAll
{
private:
    int rank;
    int worldSize;
    int tag;

    MPI_Request r_recv;
    MPI_Request* r_send;

    MPI_Status s_recv;

    int data_rcv;
    int data_snd;
    // int* data_snd;

    int flag;
    int src;


public:
    MessageChannel_OneToAll(int rank, int worldSize, int tag): rank(rank), worldSize(worldSize), tag(tag) {

        r_send = new MPI_Request[worldSize];

        MPI_Irecv(&data_rcv, 1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &r_recv);

    }

    void send(int data) {
        data_snd = data;
        //TODO: add waiting for sent request if not working

        for (size_t i = 0; i < worldSize; i++)
        {
            if(i != rank)
                MPI_Isend(&data_snd, 1, MPI_INT, i, tag, MPI_COMM_WORLD, r_send + i);
        }
    }

    int receive() {
        MPI_Test(&r_recv, &flag, &s_recv);

        if(flag == 1)
        {
            const int data_rcv_local = data_rcv;
            src = s_recv.MPI_SOURCE;

            MPI_Irecv(&data_rcv, 1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &r_recv);

            return data_rcv_local;
        }

        return -1;

    }

    int source() const
    {
        return src;
    }

    ~MessageChannel_OneToAll()
    {
        delete[] r_send;
    }
};

class MessageChannel_WorkRequest
{
private:
    MPI_Request r_send;
    MPI_Request r_recv_work_size;
    MPI_Request r_recv_parents_size;
    MPI_Request r_recv_work;
    MPI_Request r_recv_parents;

    MPI_Status  s_send;
    MPI_Status  s_recv_size;
    MPI_Status  s_recv_data;

    bool data_snd;

    std::vector<task_node_t> recv_work;

    int work_size;
    int parents_size;

    int flag;

    int requested_from;

    enum State { not_waiting_for_response,
                  waiting_for_work_size,
                  waiting_for_parents_size,
                  waiting_for_work_data,
                  waiting_for_parents_data,
                  };

    State state = State::not_waiting_for_response;

public:
    enum recv_status { nothing, data_not_ready, data_ready, src_has_no_work};

    MessageChannel_WorkRequest() {}

    void send(int dst) {
        requested_from = dst;
        if(state == 0)
        {
            MPI_Isend(&data_snd, 1, MPI_C_BOOL, dst, tag_request_work, MPI_COMM_WORLD, &r_send);
            MPI_Irecv(&work_size, 1, MPI_INT, dst, tag_response_work_size, MPI_COMM_WORLD, &r_recv_work_size);
            state = State::waiting_for_work_size;
        }
    }

    int receive(std::vector<task_node_t>& work, std::vector<unsigned int>& parents) {
        if(state == State::not_waiting_for_response)
            return recv_status::nothing;
        
        if(state == State::waiting_for_work_size)
        {
            MPI_Test(&r_recv_work_size, &flag, &s_recv_size);

            if(flag == 1)
            {
                if(work_size == 0)
                {
                    state = State::not_waiting_for_response;
                    return recv_status::src_has_no_work;
                }
                else
                {
                    // std::cout << "waiting for data " << work_size << std::endl;
                    state = State::waiting_for_work_data;
                    work.resize(work_size);
                    MPI_Irecv(work.data(), work_size, mpi_task_node_type, requested_from, tag_response_work_data, MPI_COMM_WORLD, &r_recv_work);
                }
            }
        }
        else if(state == State::waiting_for_work_data)
        {
            MPI_Test(&r_recv_work, &flag, &s_recv_data);

            if(flag == 1)
            {
                state = State::waiting_for_parents_size;
                MPI_Irecv(&parents_size, 1, MPI_INT, requested_from, tag_response_parents_size, MPI_COMM_WORLD, &r_recv_parents_size);
            }
        }
        else if(state == State::waiting_for_parents_size)
        {
            MPI_Test(&r_recv_parents_size, &flag, &s_recv_data);

            if(flag == 1)
            {
                if(parents_size == 0)
                {
                    state = State::not_waiting_for_response;
                    return recv_status::data_ready;
                }
                else
                {
                    state = State::waiting_for_parents_data;
                    parents.resize(parents_size);
                    MPI_Irecv(parents.data(), parents_size, MPI_UNSIGNED, requested_from, tag_response_parents_data, MPI_COMM_WORLD, &r_recv_parents);
                }
            }
        }
        else if(state == State::waiting_for_parents_data)
        {
            MPI_Test(&r_recv_parents, &flag, &s_recv_data);

            if(flag == 1)
            {
                state = State::not_waiting_for_response;
                return recv_status::data_ready;
            }
        }

        return recv_status::data_not_ready;
    }

    bool is_pending() const
    {
        return state != State::not_waiting_for_response;
    }

    ~MessageChannel_WorkRequest() = default;
};




class MessageChannel_WorkResponse
{
private:
    MPI_Request* r_send_work_size;
    MPI_Request* r_send_work;

    MPI_Request* r_send_parents_size;
    MPI_Request* r_send_parents;

    MPI_Request r_recv;

    MPI_Status  s_send;
    MPI_Status  s_recv;

    int worldSize;
    bool data_rcv;

    std::vector<int> work_snd_size;
    std::vector<int> parents_snd_size;
    std::vector<std::vector<task_node_t>> work_snd;
    std::vector<std::vector<unsigned int>> parents_snd;

    int sending_to;
    int flag;

public:
    MessageChannel_WorkResponse(int worldSize): worldSize(worldSize){

        sending_to = -1;

        r_send_work_size = new MPI_Request[worldSize];
        r_send_work = new MPI_Request[worldSize];
        r_send_parents_size = new MPI_Request[worldSize];
        r_send_parents = new MPI_Request[worldSize];
        work_snd_size =  std::vector<int>(worldSize);
        parents_snd_size = std::vector<int>(worldSize);
        work_snd = std::vector<std::vector<task_node_t>>(worldSize, std::vector<task_node_t>());
        parents_snd = std::vector<std::vector<unsigned int>>(worldSize);

        MPI_Irecv(&data_rcv, 1, MPI_C_BOOL, MPI_ANY_SOURCE, tag_request_work, MPI_COMM_WORLD, &r_recv);
    }

    void send_work(std::vector<task_node_t> work, std::vector<unsigned int> parents) {
        if(sending_to == -1)
            return;
        
        work_snd_size[sending_to] = work.size();
        parents_snd_size[sending_to] = parents.size();

        work_snd[sending_to] = std::vector(std::move(work));
        parents_snd[sending_to] = std::vector(std::move(parents));

        MPI_Isend(&work_snd_size[sending_to], 1, MPI_INT, sending_to, tag_response_work_size, MPI_COMM_WORLD, r_send_work_size + sending_to);
        MPI_Isend(work_snd[sending_to].data(), work_snd_size[sending_to], mpi_task_node_type, sending_to, tag_response_work_data, MPI_COMM_WORLD, r_send_work + sending_to);
        MPI_Isend(&parents_snd_size[sending_to], 1, MPI_INT, sending_to, tag_response_parents_size, MPI_COMM_WORLD, r_send_parents_size + sending_to);
        MPI_Isend(parents_snd[sending_to].data(), parents_snd_size[sending_to], MPI_UNSIGNED, sending_to, tag_response_parents_data, MPI_COMM_WORLD, r_send_parents + sending_to);

        MPI_Irecv(&data_rcv, 1, MPI_C_BOOL, MPI_ANY_SOURCE, tag_request_work, MPI_COMM_WORLD, &r_recv); // expect more work requests
        sending_to = -1;
    }

    void send_no_work() {
        if(sending_to == -1)
            return;
        
        work_snd_size[sending_to] = 0;

        MPI_Isend(&work_snd_size[sending_to], 1, MPI_INT, sending_to, tag_response_work_size, MPI_COMM_WORLD, r_send_work_size + sending_to);

        MPI_Irecv(&data_rcv, 1, MPI_C_BOOL, MPI_ANY_SOURCE, tag_request_work, MPI_COMM_WORLD, &r_recv);

        sending_to = -1;
    }

    bool anyone_asked_for_work() { 

        MPI_Test(&r_recv, &flag, &s_recv);

        if(flag == 1)
        {
            sending_to = s_recv.MPI_SOURCE;
            return true;
        }

        return false;
    }

    int requester_id() const
    {
        return sending_to;
    }


    ~MessageChannel_WorkResponse()
    {
        delete[] r_send_work_size;
        delete[] r_send_work;
        delete[] r_send_parents_size;
        delete[] r_send_parents;
    }
};

class Worker
{
private:
    const int rank;
    const int worldSize;

    bool color;
    int token;
    bool received_green_token;
    bool can_initiate_token;
    bool isIdle;

    std::deque<task_node_t> work;
    std::vector<unsigned int> parents;
    std::vector<task_t> tasks;
    std::vector<task_t> tasks_unsorted;

    
    
    //TODO: check alg vars change logic !!!!
    int c;
    const int N;
    int current_depth = 0;
    std::vector<bool> V;
    int UB;
    const int initial_UB;
    int last_no_backtrack_depth=0;

    std::vector<unsigned int> solution_id_sequence;
    int solution_cost;

    int requestWorkFrom;

    std::vector<task_node_t> received_work;

    MessageChannel_Token channel_token;
    MessageChannel_Terminate channel_terminate;
    MessageChannel_WorkRequest channel_work_request;
    MessageChannel_WorkResponse channel_work_response;
    MessageChannel_OneToAll channel_new_UB;
    MessageChannel_OneToAll channel_no_backtracking;

    
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_int_distribution<> uniform_distr;;

public:
    Worker(int rank, int worldSize, std::vector<task_t> tasks,std::vector<task_t> tasks_unsorted, int UB) : 
        rank(rank),
        worldSize(worldSize),
        color(GREEN),
        token(TOKEN_NONE),
        received_green_token(false),
        can_initiate_token(true),
        isIdle(false),
        tasks(tasks),
        tasks_unsorted(tasks_unsorted),
        N(tasks.size()),
        UB(UB),
        initial_UB(UB),
        solution_cost(UB),
        V(std::vector<bool>(N, true)),
        solution_id_sequence(std::vector<unsigned int>(N, 0)),
        channel_token(MessageChannel_Token((worldSize + rank -1)%worldSize, (rank +1)%worldSize)),
        channel_terminate(MessageChannel_Terminate(rank, worldSize)),
        channel_work_request(MessageChannel_WorkRequest()),
        channel_work_response(MessageChannel_WorkResponse(worldSize)),
        channel_new_UB(MessageChannel_OneToAll(rank, worldSize, tag_new_UB)),
        channel_no_backtracking(MessageChannel_OneToAll(rank, worldSize, tag_no_backtracking))

    {

        gen = std::mt19937(rd());
        uniform_distr = std::uniform_int_distribution<>(1, worldSize);

        requestWorkFrom = (rank + uniform_distr(gen)) %  worldSize;
    }

    ~Worker() = default;

    void set_work(std::deque<task_node_t> _work) {
        work = std::move(_work);
    }

    bool check_missed_deadline() const
    {  
        for(const auto& t : tasks)
        {
            if(V.at(t.id) and (c + t.duration > t.deadline))
            {
                return true;
            }
        }

        return false;
    }

    bool check_violated_UB() const
    {
        unsigned int sum = 0;
        unsigned int min_r = UINT_MAX;


        for(const auto& t : tasks)
        {
            if(V.at(t.id))
            {
                sum += t.duration;
                if (t.release < min_r)
                {
                    min_r = t.release;
                }
            }
        }


        const unsigned int max = min_r > c ? min_r : c;

        return (max + sum) >= UB;
    }

    // true: prune others
    // false : do not prune
    bool check_decomposition() const
    {
        unsigned int min_r = UINT_MAX;

        for(const auto& t : tasks)
        {
            if(V.at(t.id) and (t.release < min_r))
            {
                min_r = t.release;
            }
        }

        return c <= min_r;
    }



    void start() {

        while(!should_terminate())
        {
            if(work.empty())
            {
                if(!request_work())
                    isIdle = true;
            }
            else
            {
                perform_work();
            }

            if(isIdle) { 
                if(rank == 0)
                    initiate_token();
                send_token();
            }

            receive_token();
            answer_work_request();
            receive_new_UB();
            receive_no_backtracking();
        }
        // std::cout << "Terminate " << rank << std::endl;
        if(rank == 0)
        {
            channel_terminate.send();
        }
    }

    void perform_work() {
        const task_node_t tn = work.back();
        work.pop_back();

        if(current_depth > tn.depth)
        {
            // if ((current_depth - tn.depth > 2))
                // std::cout << " " << (current_depth - tn.depth) << " " << parents.size() << std::endl;
            // if((current_depth - tn.depth) > parents.size())
                // std::cout << "FAIL" << std::endl;
            for (unsigned int i = tn.depth; i < current_depth; i++) // must be always 1 ??
            {
                V[parents.back()] = true;
                parents.pop_back();
            }
        }

        current_depth = tn.depth;
        c = tn.cost;

        if(current_depth == N -1)
        {
            if(c < UB)
            {
                UB = c;
                // std::cout << rank <<  " Found solution " << UB << ". ";
                channel_new_UB.send(UB);

                for (size_t i = 0; i < N-1; i++)
                {
                    solution_id_sequence[i] = parents[i];
                    // std::cout << parents[i] << " ";
                }
                // std::cout << tn.id << std::endl;
                solution_id_sequence[N-1] = tn.id;
                solution_cost = c;
            }
        }
        else
        {
            V[tn.id] = false;

            if(check_missed_deadline() or check_violated_UB())
            {
                V[tn.id] = true;
                return;
            }

            if(check_decomposition())
            {
                last_no_backtrack_depth = tn.depth;
                send_no_backtracking();
                work.clear();
            }

            // expand node 
            bool inserted_any = false;
            for(const auto& t : tasks)
            {
                if(V[t.id])
                {
                    work.emplace_back(t.id, (c > t.release? c: t.release) + t.duration, current_depth+1);
                    inserted_any = true;
                }
            } 
            if(inserted_any)
            {
                parents.push_back(tn.id);
            }
            else
            {
                // std::cout << rank <<" inserted nothing" << std::endl;
                //TODO: think: should not really happen
            }
        }


    }

    void answer_work_request()
    {
        if(channel_work_response.anyone_asked_for_work())
        {
            if(work.size() >= 2 and work.front().depth <= SHARE_DEPTH_LIMIT)
            {
                // share_work();
                share_half_work();
            }
            else {
                channel_work_response.send_no_work();
            }
        }
    }

    void share_half_work()
    {
        if( rank > channel_work_response.requester_id())
            color = RED;

        // decide how much work to share:
        // share half of nodes of  depth above limit

        std::vector<task_node_t> work_to_share;
        work_to_share.reserve(work.size()/2);
        std::deque<task_node_t> my_new_work;
        // my_new_work.reserve(work.size());

        bool is_odd=true;
        for(const auto& tn : work)
        {
            if(is_odd and tn.depth <= SHARE_DEPTH_LIMIT)
                work_to_share.push_back(std::move(tn));
                // work_to_share.push_back(tn);
            else
                my_new_work.push_back(std::move(tn));
                // my_new_work.push_back(tn);

            // count++;
            is_odd = !is_odd;
        }
        work=std::move(my_new_work);

        const int depth_to_send = work_to_share.back().depth;
        std::vector<unsigned int> parents_to_send;

        if(depth_to_send > 0)
        {
            parents_to_send.reserve(depth_to_send);
            for (size_t i = 0; i < depth_to_send; i++)
            {
                parents_to_send.push_back( parents[i] );
            }
        }
       
        channel_work_response.send_work(std::move(work_to_share), std::move(parents_to_send));
    }


    void share_work()
    {
        if( rank > channel_work_response.requester_id())
            color = RED;

        // decide how much work to share:
        // share half of nodes of shallowest depth

        const int depth_to_send = work.front().depth;
        int count = 0;
        for(const auto& tn : work)
        {
            if(tn.depth != depth_to_send)
                break;
            count++;
        }
        const int shared_work_size = (count+1)/2;

        std::vector<task_node_t> work_to_share;
        work_to_share.reserve(shared_work_size);

        for (size_t i = 0; i < shared_work_size; i++)
        {
            work_to_share.push_back( work.front() );
            work.pop_front();
        }

        std::vector<unsigned int> parents_to_send;

        if(depth_to_send > 0)
        {
            parents_to_send.reserve(depth_to_send);
            for (size_t i = 0; i < depth_to_send; i++)
            {
                parents_to_send.push_back( parents[i] );
            }
        }
       
        channel_work_response.send_work(std::move(work_to_share), std::move(parents_to_send));
    }
        

    bool request_work() {
        // returns false if process from whom the work was requested have no work

        // return false;
        if(!channel_work_request.is_pending()) // send work request
        {
            parents.clear();
            
            received_work.clear();

            channel_work_request.send(requestWorkFrom);
            return true;
        }
        else // work request was sent, check if ready
        {
            const int resp = channel_work_request.receive(received_work, parents);
            if(resp == MessageChannel_WorkRequest::recv_status::src_has_no_work)
            {
                requestWorkFrom = (worldSize + requestWorkFrom - 1 )%worldSize;
                // requestWorkFrom = (rank + uniform_distr(gen)) %  worldSize;

                if(requestWorkFrom == rank)
                    requestWorkFrom = (worldSize + requestWorkFrom - 1 )%worldSize;

                return false;
            }
            else if (resp == MessageChannel_WorkRequest::data_ready)
            {
                // std::cout << "received" << std::endl;
                isIdle = false;
                V = std::vector<bool>(N,true);
                for (int i = received_work.size() -1 ; i >=0 ; i--)
                {
                    work.push_front(received_work[i]);
                }
                for (int i = 0 ; i < parents.size() ; i++)
                {
                    V[parents[i]] = false;
                }
                current_depth = work.back().depth;
                return true;
            }
        }

        return true;
    }

    void receive_new_UB()
    {
        while(true)
        {
            const int new_UB = channel_new_UB.receive();
            if(new_UB == -1)
                break;
            if(new_UB < UB)
            {
                UB = new_UB;
            }
            // break;
        }
    }

    void receive_no_backtracking()
    {
        while(true)
        {

            const int new_no_backtracking_depth = channel_no_backtracking.receive();
            if(new_no_backtracking_depth == -1)
            {
                break;
            }
            if(new_no_backtracking_depth > last_no_backtrack_depth)
            {
                work.clear();
                parents.clear();
                last_no_backtrack_depth = new_no_backtracking_depth;
            }
            else if(new_no_backtracking_depth == last_no_backtrack_depth)
            {
                if(rank < channel_no_backtracking.source())
                {
                    work.clear();
                    parents.clear();
                }
            }
            // break;

        }
    }

    void send_new_UB()
    {
        channel_new_UB.send(UB);
    }

    void send_no_backtracking()
    {
        // std::cout << "sending no backtrack" << std::endl;
        channel_no_backtracking.send(last_no_backtrack_depth);
    }

    void initiate_token()
    {
        if(can_initiate_token)
        {
            // std::cout << "Token initiate" << std::endl;
            token = TOKEN_GREEN;
            color = GREEN;
            can_initiate_token = false;
        }
    }

    void send_token()
    {
        if(token != TOKEN_NONE)
        {
            if(color == RED)
            {
                // std::cout << "Token send " << rank << " red " << std::endl;
                channel_token.send(TOKEN_RED);
            }
            else
            {
                // std::cout << "Token send " << rank << (token == TOKEN_RED? " red" : " green") << std::endl;
                channel_token.send(token);
            }
            token = TOKEN_NONE;
            color=GREEN;

        }
    }

    void receive_token()
    {
        if(channel_token.receive(token))
        {
            // std::cout << "Token recv " << rank << (token == TOKEN_RED? " red" : " green")  << std::endl;
            received_green_token = (token == TOKEN_GREEN);
            can_initiate_token = (token == TOKEN_RED);
        }
    }

    bool should_terminate() {
        if(rank == 0)
        {
            return received_green_token;
        }
        else
        {
            return channel_terminate.receive();
        }
    }

    void generateSolution(const std::string& outputFileName) const
    {
        int local_res[2] = {solution_cost, rank};
        int global_res[2];

        MPI_Allreduce(local_res, global_res, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        if(rank == global_res[1]) 
        {
            if(global_res[0] == initial_UB ) // no solution found
            {
                writeOutput(outputFileName,{});
            }
            else
            {
                writeOutput(outputFileName, constructSolution());
            }
        }
    }

    std::vector<unsigned int> constructSolution() const
    {
        std::vector<unsigned int> solution(N,0);

        int time = 0;
        for (auto i : solution_id_sequence)
        {
            // std::cout << i <<" ";
            time = (time > tasks_unsorted.at(i).release ? time : tasks_unsorted.at(i).release);
            solution[i] = time;
            time+=tasks_unsorted.at(i).duration;
        }
        
        // std::cout << std::endl;
        return solution;
    }

};

const auto comp_deadline_release = [ ]( const auto& lhs, const auto& rhs )
                    {
                            if(lhs.deadline == rhs.deadline)
                            {
                                return lhs.release > rhs.release;

                            }
                            return lhs.deadline > rhs.deadline;
                    };

const auto comp_release_deadline = [ ]( const auto& lhs, const auto& rhs )
                    {
                            if(lhs.release == rhs.release)
                            {
                                return lhs.deadline > rhs.deadline;

                            }
                            return lhs.release > rhs.release;
                    };

const auto comp_deadline_duration = [ ]( const auto& lhs, const auto& rhs )
                    {
                            if(lhs.deadline == rhs.deadline)
                            {
                                return lhs.duration < rhs.duration;

                            }
                            return lhs.deadline > rhs.deadline;
                    };

const auto comp_duration_deadline = [ ]( const auto& lhs, const auto& rhs )
                    {
                            if(lhs.duration == rhs.duration)
                            {
                                return lhs.deadline > rhs.deadline;

                            }
                            return lhs.duration < rhs.duration;
                    };

const std::vector<std::function<bool(const task_t &, const task_t &)>> comps = {comp_deadline_release, comp_release_deadline, comp_deadline_duration, comp_duration_deadline};

std::deque<task_node_t> get_initial_work(std::vector<task_t>& tasks, int worldSize, int rank)
{
    std::deque<task_node_t> work;

    unsigned int my_initial_tasks_num;
    unsigned int my_initial_task_id;

    unsigned int N = tasks.size();

    if ( N <= worldSize)
    {
        if (rank < N)
        {
            my_initial_tasks_num = 1;
            my_initial_task_id = rank;
        }
        else
        {
            my_initial_tasks_num = 0;
            my_initial_task_id = 0; // should not be used
        }
    }
    else
    {
        my_initial_tasks_num = N/worldSize;
        my_initial_task_id = rank* my_initial_tasks_num;

        if (rank == worldSize - 1)
        {
            my_initial_tasks_num += (N - worldSize*my_initial_tasks_num);
        }
    }
    // if(my_initial_tasks_num > 1)
        // std::sort(tasks.begin()+my_initial_task_id, tasks.begin()+my_initial_task_id+my_initial_tasks_num, comp_deadline_release);

    // std::cout << rank << " N "<< N << " my_initial_task_id " << my_initial_task_id << " last task id  " << (my_initial_task_id + my_initial_tasks_num -1)<< " my_initial_tasks_num "<< my_initial_tasks_num <<  std::endl;

    for (unsigned int i = my_initial_task_id; i < my_initial_task_id + my_initial_tasks_num; i++)
    {
        work.emplace_back(tasks.at(i).id, tasks.at(i).cost_lb, 0);
    }

    return work;
}



int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int worldSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    check_args(argc, argv, rank);

    create_task_type();
    create_task_node_type();

    std::vector<task_t> tasks;
    std::vector<task_t> tasks_unsorted;
    unsigned int UB;

    if(rank == 0)
    {
        tasks = readInstance(argv[1]);
        UB = validate(tasks);
        if(UB == 0)
        {
            writeOutput(argv[2],{});
            tasks.clear();
        }

        UB+=1;

        SHARE_DEPTH_LIMIT = tasks.size() > 5 ? ( tasks.size() - 5 ) : 0; //TODO: tune

        distributeInitialData(tasks, UB);
    }
    else
    {
        tie(tasks,UB) = receiveInitialData();
    }

    if(tasks.empty())
    {
        MPI_Finalize();
        exit(0);
    }

    std::deque<task_node_t> work = get_initial_work(tasks, worldSize, rank);
    tasks_unsorted = tasks;
    
    std::sort(tasks.begin(), tasks.end(), comps[rank%comps.size()]);

    Worker worker(rank, worldSize, tasks, tasks_unsorted, UB); //TODO tasks as reference 

    worker.set_work(std::move(work));
    worker.start();
    worker.generateSolution(argv[2]);

    MPI_Finalize();

    return 0;
}
